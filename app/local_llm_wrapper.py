"""
local_llm_wrapper.py

LangChain-compatible wrapper for running local Hugging Face Transformers models.

───────────────────────────────────────────────────────────────────────────────

Why do we need this?

LangChain supports many LLM providers like OpenAI and Hugging Face Hub (API-based),
but it does not include built-in support for models running locally via the
`transformers` library.

This wrapper bridges that gap by adapting a local Hugging Face Transformers model
to LangChain’s expected interface.

Benefits of using a local model with this wrapper:
- No API keys or internet access required
- Improved privacy: all data stays on your machine
- Zero cost per call: no billing or rate limits
- More flexibility: use any Hugging Face model (e.g. flan, Mistral, etc.)

You can plug this into LangChain chains like `RetrievalQA` exactly as you would
an API-based model; without changing your pipeline logic.
"""

from langchain.llms.base import LLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional, List, Mapping, Any
import torch
from pydantic import Field, PrivateAttr

class LocalTransformersLLM(LLM):
    """
    LangChain-compatible wrapper for a local Hugging Face Transformers Seq2Seq model.

    This class adapts a local transformer model so it behaves like an API-backed LLM
    in LangChain, allowing local inference without external API calls.

    Attributes:
        model_name (str): Hugging Face model repository identifier.
        max_length (int): Maximum tokens to generate in response.

    Usage example:
        llm = LocalTransformersLLM(model_name="google/flan-t5-base", max_length=512)
        response = llm("What is LangChain?")
    """

    # Pydantic fields - required for LangChain to serialize/validate the model config.
    model_name: str = Field(default="google/flan-t5-base", description="HF model repo id")
    max_length: int = Field(default=512, description="Maximum output token length")

    # Private attributes (not part of model init/validation)
    _tokenizer: AutoTokenizer = PrivateAttr()
    _model: AutoModelForSeq2SeqLM = PrivateAttr()
    _device: torch.device = PrivateAttr()

    def __init__(self, **kwargs):
        """
        Initialize the local Hugging Face tokenizer and model.

        Calls the parent constructor to set Pydantic-managed fields, then loads
        the tokenizer and model from Hugging Face hub.

        Moves the model to GPU if available, otherwise CPU.

        Args:
            kwargs: Should include 'model_name' and/or 'max_length' optionally.
        """
        # Let Pydantic set fields (model_name, max_length) before further init
        super().__init__(**kwargs)

        # Load tokenizer and model using the configured model_name
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Select device (GPU if available, else CPU)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Core LangChain method to generate text from the model given a prompt.

        Args:
            prompt (str): The input prompt to generate a response for.
            stop (Optional[List[str]]): Optional stop tokens (not implemented).

        Returns:
            str: The generated output text from the model.
        """
        # Tokenize input prompt and move tensors to the device
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        # Generate output tokens using the model with max_length limit
        outputs = self._model.generate(**inputs, max_length=self.max_length)

        # Decode the token IDs back into a string, skipping special tokens
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Provide identifying parameters for LangChain's internal caching, logging.

        Returns:
            dict: Model-specific metadata.
        """
        return {"model_name": self.model_name, "max_length": self.max_length}

    @property
    def _llm_type(self) -> str:
        """
        Returns a string identifying the LLM type for LangChain.

        This helps LangChain recognize this as a local transformer model.

        Returns:
            str: A string identifier for this custom LLM wrapper.
        """
        return "local_transformers"
