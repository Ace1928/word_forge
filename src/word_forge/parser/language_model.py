from typing import Any, Dict, Optional, Union

import nltk  # type: ignore
import torch
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# Download NLTK data quietly
nltk.download("wordnet", quiet=True)  # type: ignore
nltk.download("omw-1.4", quiet=True)  # type: ignore


class ModelState:
    """
    Manages transformer model state safely with lazy initialization.

    This singleton class ensures the model is only loaded when needed
    and properly configured for efficient inference.
    """

    _initialized = False
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None
    model: Optional[PreTrainedModel] = None
    model_name: str = "qwen/qwen2.5-0.5b-instruct"
    _model_name: str = "qwen/qwen2.5-0.5b-instruct"
    _device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _inference_failures: int = 0
    _max_failures: int = 5
    _failure_threshold_reached: bool = False

    @classmethod
    def get_model_name(cls) -> str:
        """Returns the configured model name."""
        return cls._model_name

    @classmethod
    def is_initialized(cls) -> bool:
        """Checks if the model and tokenizer are initialized."""
        return cls._initialized

    @classmethod
    def set_model(cls, model_name: str) -> None:
        """
        Set the model to use for text generation.

        Args:
            model_name: Name of the Hugging Face model to use
        """
        cls._model_name = model_name
        cls._initialized = False  # Force reinitialization with new model

    @classmethod
    def initialize(cls) -> bool:
        """
        Initialize the model and tokenizer if not already done.

        Returns:
            True if initialization was successful, False otherwise
        """
        if cls.is_initialized():
            return True

        if cls._failure_threshold_reached:
            print(
                f"Model initialization skipped: Failure threshold ({cls._max_failures}) reached."
            )
            return False

        try:
            # Load tokenizer with explicit type annotation
            cls.tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
            assert cls.tokenizer is not None, "Tokenizer loading returned None"

            # Load model with appropriate configuration
            cls.model = AutoModelForCausalLM.from_pretrained(
                cls._model_name,
                device_map=str(cls._device),
                torch_dtype=(
                    torch.float16 if cls._device.type == "cuda" else torch.float32
                ),
            )
            assert cls.model is not None, "Model loading returned None"

            cls._initialized = True
            print(
                f"Model '{cls._model_name}' initialized successfully on {cls._device}."
            )
            return True
        except Exception as e:
            print(f"Model initialization failed for '{cls._model_name}': {str(e)}")
            cls._inference_failures += 1
            if cls._inference_failures >= cls._max_failures:
                cls._failure_threshold_reached = True
                print(
                    f"Failure threshold ({cls._max_failures}) reached. Disabling model."
                )
            return False

    @classmethod
    def generate_text(
        cls,
        prompt: str,
        max_new_tokens: Optional[int] = 64,
        temperature: float = 0.7,
        num_beams: int = 3,
    ) -> Optional[str]:
        """
        Generate text using the loaded model.

        Args:
            prompt: Input text to generate from
            max_new_tokens: Maximum number of tokens to generate (None for model's maximum capacity)
            temperature: Sampling temperature
            num_beams: Number of beams for beam search

        Returns:
            Generated text or None if generation failed
        """
        if not cls.initialize():
            return None

        # Safety check - both must be initialized
        if cls.tokenizer is None or cls.model is None:
            print("Error: Tokenizer or model is None after initialization attempt.")
            return None

        try:
            # Create input tensors
            input_tokens = cls.tokenizer(prompt, return_tensors="pt")
            input_ids = input_tokens["input_ids"]
            input_ids = input_ids.to(cls._device)

            attention_mask = None
            if "attention_mask" in input_tokens:
                attention_mask = input_tokens["attention_mask"].to(cls._device)

            # Configure generation parameters
            gen_kwargs: Dict[str, Any] = {
                "temperature": temperature,
                "num_beams": num_beams,
                "do_sample": temperature > 0,
            }

            # Calculate max_length carefully
            input_length = input_ids.shape[1]
            model_max_length = 2048
            if hasattr(cls.model.config, "max_position_embeddings"):
                model_max_length = getattr(
                    cls.model.config, "max_position_embeddings", model_max_length
                )

            if max_new_tokens is None:
                gen_kwargs["max_length"] = model_max_length
            else:
                gen_kwargs["max_length"] = min(
                    input_length + max_new_tokens, model_max_length
                )

            # Handle pad_token_id carefully
            if cls.tokenizer.pad_token_id is None:
                if cls.tokenizer.eos_token_id is not None:
                    cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id
                    gen_kwargs["pad_token_id"] = cls.tokenizer.eos_token_id
                else:
                    print(
                        "Warning: Tokenizer has no pad_token_id or eos_token_id. Generation might be unstable."
                    )
            else:
                gen_kwargs["pad_token_id"] = cls.tokenizer.pad_token_id

            # Generate text
            with torch.no_grad():
                outputs = cls.model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
                )

            # Process the output
            output_ids = outputs[0] if outputs.ndim > 1 else outputs
            newly_generated_ids = output_ids[input_length:]
            result = cls.tokenizer.decode(newly_generated_ids, skip_special_tokens=True)

            return result.strip()

        except Exception as e:
            cls._inference_failures += 1
            if cls._inference_failures >= cls._max_failures:
                cls._failure_threshold_reached = True
                print(
                    f"Failure threshold ({cls._max_failures}) reached. Disabling model."
                )
            print(f"Text generation failed: {str(e)}", exc_info=True)
            return None

    def query(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = 256,
        temperature: float = 0.7,
        num_beams: int = 3,
    ) -> Optional[str]:
        """
        Query the model with a prompt.

        Args:
            prompt: Input text to query
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            num_beams: Number of beams for beam search

        Returns:
            Generated text or None if generation failed
        """
        return self.generate_text(prompt, max_new_tokens, temperature, num_beams)
