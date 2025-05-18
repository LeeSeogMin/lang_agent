"""Main graph builder that compiles all sub-graphs."""

import logging
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from backend.agents.orchestrator.prompts import ANSWER_PROMPT, ROUTER_SYSTEM_PROMPT
from backend.agents.orchestrator.schemas import (
    AgentState,
    AnswerReturn,
    SummaryReturn,
)
from backend.agents.knowledge.graph import create_knowledge_graph
from backend.agents.data_analysis.graph import create_data_analysis_graph
from backend.config import settings

# Initialize model for routing and summarization using shared configuration
model = settings.get_model()


# Node functions


async def summarize_conversation(state: AgentState) -> SummaryReturn:
    """Summarize older messages while keeping recent context."""
    messages = state.messages
    summary = state.summary

    # Create summarization prompt
    if summary:
        summary_prompt = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_prompt = "Create a summary of the conversation above:"

    # Add prompt to history
    messages_for_summary = messages + [HumanMessage(content=summary_prompt)]
    response = await model.ainvoke(messages_for_summary)

    # Keep only the last exchange (2 messages) and add summary
    messages_to_delete = messages[:-2] if len(messages) > 2 else []
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_delete]

    # Add summary as system message before the kept messages
    summary_msg = SystemMessage(
        content=f"Previous conversation summary: {response.content}"
    )

    # Return SummaryReturn object
    return SummaryReturn(
        summary=response.content,
        messages=delete_messages + [summary_msg],
    )


async def route_message(
    state: AgentState,
) -> Command[Literal["answer", "knowledge", "data_analyzer"]]:
    """Router that decides if we need to delegate to specialized agents."""
    messages = state.messages
    if not messages:
        return Command(
            goto="knowledge",
            update={"routing_decision": "Default to knowledge (no messages)"},
        )

    try:
        last_message_content = messages[-1].content
        analyze_prefix = "ANALYZE DATA:"

        if isinstance(last_message_content, str) and analyze_prefix in last_message_content:
            print(f"[DEBUG] Data analysis prefix detected in message.")
            # Extract the data content from the message
            data_parts = last_message_content.split(analyze_prefix, 1)
            if len(data_parts) > 1:
                data_to_analyze = data_parts[1].strip()
                print(f"[DEBUG] Data extracted for analysis. Length: {len(data_to_analyze)} chars")
                return Command(
                    goto="data_analyzer",
                    update={
                        "document_content": data_to_analyze,
                        "routing_decision": "Routing to data analyzer due to ANALYZE DATA prefix.",
                        "knowledge_findings": None,
                        "data_analysis_response": None,
                    },
                )
            else:
                print(f"[DEBUG] Failed to extract data after prefix.")

        # Get recent context for LLM-based routing if prefix not found
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        recent_content = "\n".join(
            [f"{msg.__class__.__name__}: {msg.content}" for msg in recent_messages]
        )

        # Check for manual data analysis triggers
        data_analysis_triggers = [
            "분석해", "analyze", "분석 요청", "데이터 분석", "파일 분석", "통계", "statistics", 
            "csv 분석", "텍스트 분석", "데이터를 분석", "파일을 분석"
        ]
        
        if isinstance(last_message_content, str) and any(trigger in last_message_content.lower() for trigger in data_analysis_triggers):
            print(f"[DEBUG] Data analysis keyword detected in message: {last_message_content}")
            
            # Check if we have uploaded data in session
            if hasattr(state, "uploaded_data_file") and state.uploaded_data_file:
                print(f"[DEBUG] Using uploaded data file: {state.uploaded_data_file}")
                # Read file content
                try:
                    with open(state.uploaded_data_file, "r", encoding="utf-8") as f:
                        file_content = f.read()
                    print(f"[DEBUG] Successfully read file with {len(file_content)} chars")
                    return Command(
                        goto="data_analyzer",
                        update={
                            "document_content": file_content,
                            "routing_decision": "Routing to data analyzer due to keyword detection with uploaded file.",
                            "knowledge_findings": None,
                            "data_analysis_response": None,
                        },
                    )
                except Exception as e:
                    print(f"[DEBUG] Error reading file: {str(e)}")

        # Get routing decision with explicit encoding handling
        response = await model.ainvoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT.format(context=recent_content)),
                HumanMessage(
                    content=messages[-1].content.encode('utf-8').decode('utf-8')
                ),  # Ensure proper UTF-8 encoding for Korean text
            ]
        )

        # Get the route from the response with error handling
        try:
            next_step_str = (
                response.content.split("[Selected Route]")[1].split("\n")[1].strip().upper()
            )
            print(f"[DEBUG] Router selected route: {next_step_str}")
        except Exception as e:
            logging.warning(f"Error parsing routing response: {e}. Using default route.")
            next_step_str = "KNOWLEDGE"  # Default to knowledge on parsing error
    except Exception as e:
        logging.error(f"Error in route_message: {e}", exc_info=True)
        return Command(
            goto="answer",
            update={
                "routing_decision": "Error in routing, defaulting to direct answer",
                "error": str(e)
            },
        )

    # Map to valid next steps
    route_mapping = {
        "ANSWER": "answer",
        "KNOWLEDGE": "knowledge",
        "ANALYZE": "data_analyzer",
    }

    # Determine the actual next node name
    next_node_name = route_mapping.get(next_step_str, "knowledge")

    # Return Command to transition and update state
    return Command(goto=next_node_name, update={"routing_decision": response.content})


# Edge conditions


def should_summarize_conversation(state: AgentState) -> bool:
    """Check if we should summarize the conversation."""
    messages = state.messages

    # First check message count - summarize if more than threshold
    if len(messages) > 10:  # Only summarize after 10 messages
        # Only summarize after an AI response (complete exchange)
        last_message = messages[-1]
        return isinstance(last_message, AIMessage)
    return False


async def answer(state: AgentState) -> AnswerReturn:
    """Generate a direct answer to the user's question from conversation context."""
    messages = state.messages
    routing_decision = state.routing_decision

    # Check LLM model configuration
    if model is None:
        print("[DEBUG] LLM model is None!")
        return AnswerReturn(
            messages=[AIMessage(content="LLM model is not configured correctly. Please check API key/model settings.")]
        )

    try:
        # Determine which agent output we're using (if any)
        context = ""
        route = state.routing_decision.split(" ")[0].upper() if state.routing_decision else "ANSWER"

        print(f"[DEBUG] answer() called. route: {route}")
        print(f"[DEBUG] state.routing_decision: {state.routing_decision}")
        print(f"[DEBUG] state.error: {getattr(state, 'error', None)}")
        print(f"[DEBUG] state.knowledge_findings: {getattr(state, 'knowledge_findings', None)}")
        print(f"[DEBUG] state.data_analysis_response: {getattr(state, 'data_analysis_response', None)}")
        
        # Check if this might be a data analysis request that wasn't properly routed
        if route != "ANALYZE" and hasattr(state, "messages") and state.messages:
            last_message = state.messages[-1].content if state.messages[-1] and hasattr(state.messages[-1], "content") else ""
            if isinstance(last_message, str) and "ANALYZE DATA:" in last_message:
                print(f"[DEBUG] Found ANALYZE DATA prefix in message but route is {route}. Forcing data analysis route.")
                route = "ANALYZE"
                
                # Extract the data for analysis
                data_parts = last_message.split("ANALYZE DATA:", 1)
                if len(data_parts) > 1:
                    data_to_analyze = data_parts[1].strip()
                    # Create a temporary DataAnalysisState to process the data
                    from backend.agents.data_analysis.graph import create_data_analysis_graph
                    from backend.agents.data_analysis.schemas import DataAnalysisState, DataAnalysisResponse
                    
                    data_analysis_state = DataAnalysisState(
                        document_content=data_to_analyze,
                        messages=state.messages,
                        uploaded_data_file=getattr(state, "uploaded_data_file", None)
                    )
                    
                    try:
                        # Run data analysis directly
                        data_analysis_graph = create_data_analysis_graph().compile()
                        data_analysis_result = await data_analysis_graph.ainvoke(data_analysis_state)
                        
                        print(f"[DEBUG] Emergency data analysis complete. Result type: {type(data_analysis_result)}")
                        
                        # Handle result in different data structures
                        if hasattr(data_analysis_result, "data_analysis_response"):
                            # Direct attribute access
                            data_analysis_response = data_analysis_result.data_analysis_response
                            print(f"[DEBUG] Retrieved data_analysis_response directly as attribute")
                        elif isinstance(data_analysis_result, dict) and "data_analysis_response" in data_analysis_result:
                            # Dictionary access
                            data_analysis_response = data_analysis_result["data_analysis_response"]
                            print(f"[DEBUG] Retrieved data_analysis_response from dictionary")
                        elif hasattr(data_analysis_result, "get") and data_analysis_result.get("data_analysis_response"):
                            # Object with get method
                            data_analysis_response = data_analysis_result.get("data_analysis_response")
                            print(f"[DEBUG] Retrieved data_analysis_response using get() method")
                        else:
                            # Fallback: assume the entire result is the response
                            data_analysis_response = data_analysis_result
                            print(f"[DEBUG] Using entire result as data_analysis_response")
                        
                        # Create a proper DataAnalysisResponse object
                        if not isinstance(data_analysis_response, DataAnalysisResponse):
                            # Convert dictionary to DataAnalysisResponse if needed
                            if isinstance(data_analysis_response, dict):
                                formatted_analyses = data_analysis_response.get("formatted_segment_analyses", "")
                                segment_analyses = data_analysis_response.get("segment_analyses", [])
                                num_segments = data_analysis_response.get("num_segments", 0)
                                metadata = data_analysis_response.get("metadata", {})
                            else:
                                # Extract attributes from object
                                formatted_analyses = getattr(data_analysis_response, "formatted_segment_analyses", "")
                                segment_analyses = getattr(data_analysis_response, "segment_analyses", [])
                                num_segments = getattr(data_analysis_response, "num_segments", 0)
                                metadata = getattr(data_analysis_response, "metadata", {})
                            
                            state.data_analysis_response = DataAnalysisResponse(
                                formatted_segment_analyses=formatted_analyses,
                                segment_analyses=segment_analyses,
                                num_segments=num_segments,
                                metadata=metadata
                            )
                            print(f"[DEBUG] Created new DataAnalysisResponse object")
                        else:
                            # Use as is
                            state.data_analysis_response = data_analysis_response
                            print(f"[DEBUG] Used DataAnalysisResponse directly")
                    except Exception as e:
                        print(f"[DEBUG] Error in emergency data analysis: {str(e)}")
                        import traceback
                        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                        return AnswerReturn(
                            messages=[AIMessage(content=f"데이터 분석 중 오류가 발생했습니다: {str(e)}")]
                        )

        # Handle routing error if present
        if hasattr(state, 'error') and state.error:
            print(f"[DEBUG] Routing error present: {state.error}")
            return AnswerReturn(
                messages=[AIMessage(content=f"Sorry, an error occurred while processing your message: {state.error}")]
            )

        # Case 1: Knowledge agent output (from knowledge node)
        if route == "KNOWLEDGE" and state.knowledge_findings:
            knowledge_data = state.knowledge_findings

            # Use the pre-formatted context if available
            if "formatted_context" in knowledge_data:
                formatted_context = knowledge_data.get("formatted_context", "")
                context = f"Context from Knowledge Agent:\n{formatted_context}\n"

        # Case 2: Data Analysis agent output (from data_analyzer node)
        elif route == "ANALYZE" and state.data_analysis_response:
            analysis_data = state.data_analysis_response
            
            # Print debug info about data analysis response
            print(f"[DEBUG] Data analysis response type: {type(analysis_data)}")
            
            # Safe attribute access with proper error handling
            formatted_analyses = ""
            if hasattr(analysis_data, "formatted_segment_analyses"):
                formatted_analyses = analysis_data.formatted_segment_analyses
                print(f"[DEBUG] Has formatted_segment_analyses: {bool(formatted_analyses)}")
                print(f"[DEBUG] Length: {len(formatted_analyses) if formatted_analyses else 0}")
            elif isinstance(analysis_data, dict) and "formatted_segment_analyses" in analysis_data:
                formatted_analyses = analysis_data["formatted_segment_analyses"]
            
            if formatted_analyses:
                # Create a rich context for the LLM
                context = f"## Data Analysis Results\n\n{formatted_analyses}\n\n"
                
                # Add metadata if available
                metadata = {}
                if hasattr(analysis_data, "metadata") and analysis_data.metadata:
                    metadata = analysis_data.metadata
                elif isinstance(analysis_data, dict) and "metadata" in analysis_data:
                    metadata = analysis_data["metadata"]
                
                # Format timestamps nicely if present
                if metadata:
                    timestamp = metadata.get("timestamp", "")
                    if timestamp:
                        try:
                            # Try to parse and format the timestamp
                            from datetime import datetime
                            dt = datetime.fromisoformat(timestamp)
                            timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            # Use as is if parsing fails
                            pass
                    
                    context += f"\n## Analysis Metadata\n"
                    context += f"- Analysis Time: {timestamp}\n"
                    context += f"- Segments Analyzed: {metadata.get('num_segments', 0)}\n"
            else:
                # Fallback if formatted analysis is not available
                segment_analyses = []
                if hasattr(analysis_data, "segment_analyses"):
                    segment_analyses = analysis_data.segment_analyses
                elif isinstance(analysis_data, dict) and "segment_analyses" in analysis_data:
                    segment_analyses = analysis_data["segment_analyses"]
                
                formatted_analyses = "\n\n".join([f"Analysis {i+1}:\n{segment}" for i, segment in enumerate(segment_analyses)])
                context = f"Data Analysis Results:\n\n{formatted_analyses}\n\n"
                
        # Get the user's question
        user_question = ""
        if messages and len(messages) > 0:
            user_question = messages[-1].content
            # Strip ANALYZE DATA: prefix if present
            if isinstance(user_question, str) and "ANALYZE DATA:" in user_question:
                user_question = "데이터를 분석해주세요."

        # Combine everything for the final prompt
        if context:
            prompt = ANSWER_PROMPT.format(
                context=context,
                question=user_question
            )
        else:
            # No specialized agent was used, just answer directly
            prompt = f"사용자의 질문에 직접 답변해주세요. 질문: {user_question}"

        # Get final answer from LLM
        response = await model.ainvoke([SystemMessage(content=prompt)])
        
        # If this was a data analysis request, add an introductory sentence
        if route == "ANALYZE":
            prefix = "여기 데이터 분석 결과입니다:\n\n"
            return AnswerReturn(messages=[AIMessage(content=f"{prefix}{response.content}")])
        
        return AnswerReturn(messages=[AIMessage(content=response.content)])

    except Exception as e:
        logging.exception(f"Error in answer function: {e}")
        return AnswerReturn(
            messages=[AIMessage(content=f"요청 처리 중 오류가 발생했습니다: {str(e)}. 다시 시도하거나 질문을 다르게 해보세요.")]
        )


# Create the main graph


def create_graph() -> Any:
    """Create and compile the complete agent graph with all components."""

    # Initialize the graph
    workflow = StateGraph(AgentState)

    # Create and compile sub-graphs
    knowledge_graph = create_knowledge_graph().compile()
    data_analysis_graph = create_data_analysis_graph().compile()

    # Add all nodes
    workflow.add_node("router", route_message)
    workflow.add_node("answer", answer)
    workflow.add_node("knowledge", knowledge_graph)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_node("data_analyzer", data_analysis_graph)

    # Start directly with router
    workflow.add_edge(START, "router")

    # Conditional edges from knowledge and data_analyzer nodes to chat history summarization or directly to answer
    workflow.add_conditional_edges(
        "knowledge",
        should_summarize_conversation,
        {True: "summarize_conversation", False: "answer"},
    )
    workflow.add_conditional_edges(
        "data_analyzer",
        should_summarize_conversation,
        {True: "summarize_conversation", False: "answer"},
    )

    # After chat history summarization, go to answer
    workflow.add_edge("summarize_conversation", "answer")

    # Router can also go directly to answer for simple questions not needing other agents
    # The answer node is the main one that ends the flow
    workflow.add_edge("answer", END)

    return workflow.compile()
