from typing import Any, Dict, Optional, Union

import nltk
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# Download NLTK data quietly
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class ModelState:
    """
    Manages transformer model state safely with lazy initialization.

    This singleton class ensures the model is only loaded when needed
    and properly configured for efficient inference.
    """

    _initialized = False
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None
    model: Optional[PreTrainedModel] = None
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
            # Load tokenizer without unnecessary cast
            cls.tokenizer = AutoTokenizer.from_pretrained(cls._model_name)

            # Load model with appropriate configuration
            cls.model = AutoModelForCausalLM.from_pretrained(
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
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        num_beams: int = 3,
    ) -> Optional[str]:
        """
        Generate text using the loaded model.

        Args:
            prompt: Input text to generate from
            max_new_tokens: Maximum number of tokens to generate
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
            input_ids = input_tokens["input_ids"].to(cls._device)

            # Handle attention mask if present
            attention_mask = None
            if "attention_mask" in input_tokens:
                attention_mask = input_tokens["attention_mask"].to(cls._device)

            # Configure generation parameters
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "num_beams": num_beams,
                "do_sample": temperature > 0.1,
            }

            # Set pad_token_id if eos_token_id exists
            if (
                hasattr(cls.tokenizer, "eos_token_id")
                and cls.tokenizer.eos_token_id is not None
            ):
                gen_kwargs["pad_token_id"] = cls.tokenizer.eos_token_id

            # Generate text
            outputs = cls.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )

            # Process the output
            result = ""
            if isinstance(outputs, torch.Tensor):
                result = cls.tokenizer.decode(
                    outputs[0].tolist(), skip_special_tokens=True
                )
            else:
                # For other output types
                result = cls.tokenizer.decode(outputs[0], skip_special_tokens=True)

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


def main():
    # Example usage
    model_state = ModelState()
    generated_text = model_state.generate_text("Once upon a time")
    print(generated_text)


if __name__ == "__main__":
    main()
