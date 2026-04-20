"""
LLM Manager — loads the GGUF model once at startup
and exposes a generate() method for chat completions.

Prompt format (from training notebook):
    ### Instruction:
    {question}

    ### Response:
    {answer}
"""

from llama_cpp import Llama
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class LLMManager:
    """Singleton manager for the loaded GGUF model."""

    def __init__(self):
        self.model: Llama | None = None
        self.loaded: bool = False

    def load(self):
        """Load the GGUF model from disk. Called once at app startup."""
        try:
            logger.info(f"Loading model from: {settings.model_path}")

            self.model = Llama(
                model_path=settings.model_path,
                n_ctx=settings.n_ctx,
                n_threads=settings.n_threads,
                n_gpu_layers=settings.n_gpu_layers,
                verbose=False,
            )

            self.loaded = True
            logger.info("Model loaded successfully!")

        except FileNotFoundError:
            logger.error(f"Model file not found: {settings.model_path}")
            logger.error("Place your .gguf file inside the models/ folder")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(self, messages: list[dict]) -> str:
        """
        Generate a response given a list of chat messages.

        Training format (Alpaca-style):
            ### Instruction:
            {question}

            ### Response:
            {answer}
        """
        if not self.loaded or self.model is None:
            raise RuntimeError("Model is not loaded.")

        prompt = self._build_prompt(messages)

        output = self.model(
            prompt,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            stop=["### Instruction:", "###"],
            echo=False,
        )

        reply = output["choices"][0]["text"].strip()
        return reply or "I'm sorry, I could not generate a response. Please try again."

    def _build_prompt(self, messages: list[dict]) -> str:
        """
        Build Alpaca-style prompt matching the training format.

        Single turn:
            ### Instruction:
            {question}

            ### Response:

        Multi-turn (conversation history included):
            ### Instruction:
            {q1}

            ### Response:
            {a1}

            ### Instruction:
            {q2}

            ### Response:
        """
        prompt = ""

        for msg in messages:
            role    = msg["role"]
            content = msg["content"].strip()

            if role == "user":
                prompt += f"### Instruction:\n{content}\n\n### Response:\n"
            elif role == "assistant":
                prompt += f"{content}\n\n"

        return prompt


# Single global instance — imported by routes
llm_manager = LLMManager()