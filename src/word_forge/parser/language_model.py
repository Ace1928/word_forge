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
        if cls._initialized:
            return True

        if cls._failure_threshold_reached:
            return False

        try:
            # Load tokenizer with explicit type annotation
            cls.tokenizer = AutoTokenizer.from_pretrained(cls._model_name)  # type: ignore
            assert cls.tokenizer is not None

            # Load model with appropriate configuration
            cls.model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                cls._model_name,
                device_map=cls._device,
                torch_dtype=(
                    torch.float16 if cls._device.type == "cuda" else torch.float32
                ),
            )
            cls._initialized = True
            return True
        except Exception as e:
            print(f"Model initialization failed: {str(e)}")
            cls._inference_failures += 1
            if cls._inference_failures >= cls._max_failures:
                cls._failure_threshold_reached = True
            return False

    @classmethod
    def generate_text(
        cls,
        prompt: str,
        max_new_tokens: Optional[int] = None,
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
            return None

        try:
            # Create input tensors
            input_tokens = cls.tokenizer(prompt, return_tensors="pt")
            input_ids = input_tokens["input_ids"]
            input_ids = (
                input_ids.to(cls._device)
                if isinstance(input_ids, torch.Tensor)
                else torch.tensor(input_ids, device=cls._device)
            )

            # Handle attention mask if present
            attention_mask = None
            if "attention_mask" in input_tokens:
                attention_mask = input_tokens["attention_mask"]
                attention_mask = (
                    attention_mask.to(cls._device)
                    if isinstance(attention_mask, torch.Tensor)
                    else torch.tensor(attention_mask, device=cls._device)
                )

            # Configure generation parameters
            gen_kwargs: Dict[str, Any] = {
                "temperature": temperature,
                "num_beams": num_beams,
            }
            # Use model's max length if max_new_tokens is None
            if max_new_tokens is None:
                if hasattr(cls.model.config, "max_position_embeddings"):  # type: ignore
                    # Account for context length by subtracting input length
                    input_length = input_ids.shape[1]
                    model_max_length = getattr(
                        cls.model.config, "max_position_embeddings", 2048  # type: ignore
                    )
                    gen_kwargs["max_new_tokens"] = model_max_length - input_length
                else:
                    # Fallback to a reasonable default if max length not specified
                    gen_kwargs["max_new_tokens"] = 1024
            else:
                gen_kwargs["max_new_tokens"] = max_new_tokens

            # Set pad_token_id if eos_token_id exists and is usable
            if (
                hasattr(cls.tokenizer, "eos_token_id")
                and getattr(cls.tokenizer, "eos_token_id", None) is not None
            ):
                # Handle different possible types of eos_token_id
                eos_token_id: Any = cls.tokenizer.eos_token_id  # type: ignore
                if isinstance(eos_token_id, int):
                    gen_kwargs["pad_token_id"] = eos_token_id
                elif isinstance(eos_token_id, str) and eos_token_id.isdigit():
                    gen_kwargs["pad_token_id"] = int(eos_token_id)
                elif (
                    isinstance(eos_token_id, list)
                    and eos_token_id
                    and isinstance(eos_token_id[0], int)
                ):
                    gen_kwargs["pad_token_id"] = eos_token_id[0]

            # Generate text
            outputs = cls.model.generate(  # type: ignore
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )

            # Process the output
            result = ""
            if isinstance(outputs, torch.Tensor):
                output_ids = outputs[0].tolist()  # type: ignore[attr-defined]
                result = cls.tokenizer.decode(  # type: ignore
                    output_ids, skip_special_tokens=True
                )
            else:
                # For other output types
                result = cls.tokenizer.decode(outputs[0], skip_special_tokens=True)  # type: ignore

            # Remove the prompt from the result
            if result.startswith(prompt):
                result = result[len(prompt) :]

            return result.strip()

        except Exception as e:
            cls._inference_failures += 1
            if cls._inference_failures >= cls._max_failures:
                cls._failure_threshold_reached = True
            print(f"Text generation failed: {str(e)}")
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


def main():
    # Example usage
    model_state = ModelState()
    generated_text = model_state.generate_text(
        "Create an absolutely unhinged finish to this, keep it brief: Once upon a time",
        max_new_tokens=128,
    )
    print(generated_text)


if __name__ == "__main__":
    main()
