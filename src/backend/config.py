"""Shared configuration settings for the multi-agent system."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

# Define project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Environment configuration with validation."""

    # OpenAI
    openai_api_key: str = Field(default="", description="OpenAI API key")
    
    # Anthropic
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # Tavily
    tavily_api_key: str = Field(default="", description="Tavily API key")

    # LangChain
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", description="LangSmith endpoint"
    )
    langsmith_tracing_v2: bool = Field(
        default=True, description="Enable LangSmith tracing v2"
    )
    langsmith_project: str = Field(
        default="multi_agent_system", description="LangSmith project name"
    )

    # Storage Paths
    data_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data",
        description="Base directory for data storage",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def chroma_path(self) -> Path:
        """Full path for Chroma DB storage"""
        return self.data_dir / "chroma_db"

    def ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        dirs = [
            self.data_dir,
            self.chroma_path,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load_env(cls) -> "Settings":
        """Load configuration from environment variables."""
        config = cls()
        import os
        print("[DEBUG] OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
        print("[DEBUG] ANTHROPIC_API_KEY:", os.getenv("ANTHROPIC_API_KEY"))
        config.ensure_dirs()
        return config

    # Shared model configuration
    def get_model(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        streaming: bool = True,
        **kwargs,
    ) -> ChatOpenAI | ChatAnthropic:
        """Get a configured model instance with standard settings.

        Args:
            model_name: Name of the model to use (default: gpt-4o-mini)
            temperature: Temperature setting (default: 0.7)
            streaming: Whether to enable streaming (default: True)
            **kwargs: Additional model configuration options

        Returns:
            Configured ChatOpenAI instance (or ChatAnthropic if specified)

        Raises:
            ValueError: If required API key is not set
        """
        # 모델명이 claude로 시작하는 경우에만 Anthropic API 사용
        if model_name.startswith("claude"):
            # Use Anthropic API
            if not self.anthropic_api_key:
                raise ValueError(
                    "Anthropic API key is not set. Please set the ANTHROPIC_API_KEY environment variable."
                )
                
            return ChatAnthropic(
                api_key=self.anthropic_api_key, 
                model=model_name,
                temperature=temperature,
                streaming=streaming,
                **kwargs
            )
        else:
            # Use OpenAI API as default
            if not self.openai_api_key:
                raise ValueError(
                    "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
                )
            
            # 단순화된 OpenAI 모델 초기화 (성공 사례와 유사하게)
            return ChatOpenAI(
                api_key=self.openai_api_key,
                model=model_name,
                temperature=temperature,
                timeout=60
            )


    def get_embeddings(
        self, model_name: str = "text-embedding-3-small", **kwargs
    ) -> OpenAIEmbeddings:
        """Get a configured embeddings model instance.

        Args:
            model_name: Name of the embeddings model to use (default: text-embedding-3-small)
            **kwargs: Additional embeddings configuration options

        Returns:
            Configured OpenAIEmbeddings instance
        """
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
            )
        
        # 단순화된 임베딩 모델 초기화
        return OpenAIEmbeddings(
            api_key=self.openai_api_key,
            model=model_name,
            **kwargs
        )


# Load environment configuration
settings = Settings.load_env()
