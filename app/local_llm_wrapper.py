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
- More flexibility: use any Hugging Face model (e.g. flan, Mistral, GPT-Neo, etc.)

You can plug this into LangChain chains like `RetrievalQA` exactly as you would
an API-based model; without changing your pipeline logic.
"""
import subprocess

from langchain.llms.base import LLM
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    PreTrainedModel,
)
from typing import Optional, List, Mapping, Any
import torch
from pydantic import Field, PrivateAttr


class LocalTransformersLLM(LLM):
    """
    LangChain-compatible wrapper for a local Hugging Face Transformers model.

    Supports both Seq2Seq (encoder-decoder) and causal (decoder-only) models.
    Automatically detects model architecture and loads the appropriate class.

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
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Nucleus sampling top_p")
    do_sample: bool = Field(default=True, description="Whether to sample or do greedy decoding")
    num_beams: int = Field(default=1, description="Number of beams for beam search")
    use_llamacpp: bool = Field(default=False, description="Whether to use llama.cpp backend")

    # Private attributes (not part of model init/validation)
    _tokenizer: AutoTokenizer = PrivateAttr()
    _model: PreTrainedModel = PrivateAttr()
    _device: torch.device = PrivateAttr()
    _is_seq2seq: bool = PrivateAttr()

    def __init__(self, model_name: str, use_llamacpp: bool = None, **kwargs):
        """
        Initialize the local Hugging Face tokenizer and model.

        Calls the parent constructor to set Pydantic-managed fields, then loads
        the tokenizer and model from Hugging Face hub.

        Moves the model to GPU if available, otherwise CPU.

        Args:
            kwargs: Should include 'model_name' and/or 'max_length' optionally.
        """
        super().__init__(**kwargs)
        if use_llamacpp is None:
            use_llamacpp = model_name.strip().endswith(".gguf")
        self.use_llamacpp = use_llamacpp

        if self.use_llamacpp:
            self._is_seq2seq = False
            # No need to load HF model if llama.cpp is used
            return

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Determine model type (Seq2Seq or causal) by trying to load Seq2Seq first
        try:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._is_seq2seq = True
        except Exception:
            # Fallback: try causal model loading
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                               low_cpu_mem_usage=True,
                                                               device_map="auto",  # uses CPU if no CUDA
                                                               torch_dtype="auto"
                                                               )
            self._is_seq2seq = False

        # Select device (GPU if available, else CPU)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if self.use_llamacpp:
            return self._call_llamacpp(prompt)
        return self._call_transformers(prompt)

    def _call_llamacpp(self, prompt: str) -> str:
        """
        Calls llama.cpp binary from subprocess.
        Assumes `llama` binary is available in PATH or working directory.

        Returns:
            str: Model output.
        """
        llama_bin = "/model"  # Adjust path if needed
        model_path = self.model_name  # This should point to a .gguf model file
        n_predict = str(self.max_length)

        args = [
            llama_bin,
            "-m", model_path,
            "-p", prompt,
            "-n", n_predict,
            "--temp", str(self.temperature),
            "--top-p", str(self.top_p)
        ]

        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"llama.cpp failed: {e.stderr}") from e

    def _call_transformers(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Core LangChain method to generate text from the model given a prompt.

        Args:
            prompt (str): The input prompt to generate a response for.
            stop (Optional[List[str]]): Optional stop tokens (not implemented).

        Returns:
            str: The generated output text from the model.
        """
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        if self._is_seq2seq:
            outputs = self._model.generate(
                **inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                early_stopping=True,
                eos_token_id=self._tokenizer.eos_token_id,
                # Stop generation when the "end-of-sequence" token is produced.
                pad_token_id=self._tokenizer.pad_token_id  # Use padding token ID to handle shorter sequences if needed.
            )
        else:
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                early_stopping=True,
                eos_token_id=self._tokenizer.eos_token_id,
                # Stop generation when the "end-of-sequence" token is produced.
                pad_token_id=self._tokenizer.pad_token_id  # Use padding token ID to handle shorter sequences if needed.

            )

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Provide identifying parameters for LangChain's internal caching, logging.

        Returns:
            dict: Model-specific metadata.
        """
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "is_seq2seq": self._is_seq2seq,
            "use_llamacpp": self.use_llamacpp
        }

    @property
    def _llm_type(self) -> str:
        """
        Returns a string identifying the LLM type for LangChain.

        This helps LangChain recognize this as a local transformer model.

        Returns:
            str: A string identifier for this custom LLM wrapper.
        """
        return "llama.cpp" if self.use_llamacpp else "local_transformers"

