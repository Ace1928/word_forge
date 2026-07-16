"""Lazy, resource-conscious local language-model inference.

The lexical core must remain fast and usable without PyTorch or Transformers.
This module therefore imports the optional inference stack only when a caller
actually initializes a model, not when :class:`ModelState` is imported or
constructed.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import threading
from os import PathLike
from typing import Any, Dict, Optional, Tuple, Union, cast

from word_forge.parser.model_profiles import DEFAULT_MODEL_ID

logger = logging.getLogger(__name__)

MAX_AUTOMATIC_NEW_TOKENS = 256


class ModelDependencyError(RuntimeError):
    """Raised when the optional local-inference stack is unavailable."""


def _load_inference_stack() -> Tuple[Any, Any, Any]:
    """Import and return Torch plus the two Transformers auto classes.

    Keeping this operation behind an explicit function prevents a normal
    lexical-only process from paying the substantial import-time memory and
    latency cost of the model stack.
    """

    try:
        torch_module = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")
        tokenizer_class = transformers_module.AutoTokenizer
        model_class = transformers_module.AutoModelForCausalLM
    except (AttributeError, ImportError, OSError) as exc:
        raise ModelDependencyError(
            "Local model inference requires the 'llm' extra; install "
            "word_forge[llm]."
        ) from exc
    return torch_module, tokenizer_class, model_class


class ModelState:
    """Own a lazily initialized language model and tokenizer.

    Instances are independent and safe to share across worker threads. Model
    loading is serialized so concurrent first-use calls cannot allocate the
    same weights multiple times, and generation is serialized because many
    Transformers backends do not guarantee concurrent mutation-free inference.

    Args:
        model_name: Hugging Face model identifier or local model path.
        device: Optional Torch device or device name. When omitted, CUDA is
            selected when available and CPU otherwise.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_ID,
        device: Optional[Any] = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name.strip()
        self._requested_device = device
        self.device: Any = device if device is not None else "uninitialized"
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self._torch: Optional[Any] = None
        self._initialized = False
        self._inference_failures = 0
        self._max_failures = 5
        self._failure_threshold_reached = False
        self._last_error: Optional[str] = None
        self._initialization_lock = threading.Lock()
        self._concurrency_limit = 4
        self._generation_semaphore = threading.BoundedSemaphore(value=self._concurrency_limit)

    @property
    def last_error(self) -> Optional[str]:
        """Return the latest initialization or generation error, if any."""

        return self._last_error

    def get_model_name(self) -> str:
        """Return the configured model name."""

        return self.model_name

    def is_initialized(self) -> bool:
        """Return whether both model and tokenizer are ready for inference."""

        return (
            self._initialized and self.model is not None and self.tokenizer is not None
        )

    def set_model(self, model_name: str) -> None:
        """Select a different model and release the current model resources."""

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        for _ in range(self._concurrency_limit):
            self._generation_semaphore.acquire()
        try:
            with self._initialization_lock:
                self._release_resources()
                self.model_name = model_name.strip()
                self._inference_failures = 0
                self._failure_threshold_reached = False
                self._last_error = None
        finally:
            for _ in range(self._concurrency_limit):
                self._generation_semaphore.release()

    def initialize(self) -> bool:
        """Load the configured tokenizer and model on first use.

        Returns:
            ``True`` when inference is ready, otherwise ``False``. Failure
            details are available through :attr:`last_error` and logging.
        """

        if self.is_initialized():
            return True
        with self._initialization_lock:
            if self.is_initialized():
                return True
            if self._failure_threshold_reached:
                logger.error(
                    "Model initialization disabled after %d failures",
                    self._max_failures,
                )
                return False
            return self._initialize_unlocked()

    def _initialize_unlocked(self) -> bool:
        """Initialize model resources while the initialization lock is held."""

        try:
            torch_module, tokenizer_class, model_class = _load_inference_stack()
            self._torch = torch_module
            if self._requested_device is None:
                self.device = torch_module.device(
                    "cuda" if torch_module.cuda.is_available() else "cpu"
                )
            else:
                self.device = torch_module.device(self._requested_device)

            if hasattr(self.device, "type") and self.device.type == "cpu":
                torch_module.set_num_threads(1)

            tokenizer = tokenizer_class.from_pretrained(
                cast(Union[str, PathLike[str]], self.model_name)
            )
            if tokenizer is None:
                raise RuntimeError("Tokenizer loading returned no tokenizer")

            device_type = getattr(self.device, "type", str(self.device))
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": (
                    torch_module.float16
                    if device_type in {"cuda", "mps"}
                    else torch_module.float32
                )
            }
            if device_type == "cuda" and importlib.util.find_spec("accelerate"):
                model_kwargs["device_map"] = "auto"

            model = model_class.from_pretrained(
                cast(Union[str, PathLike[str]], self.model_name), **model_kwargs
            )
            if model is None:
                raise RuntimeError("Model loading returned no model")
            if model_kwargs.get("device_map") != "auto":
                model.to(self.device)
            if hasattr(model, "eval"):
                model.eval()

            self.tokenizer = tokenizer
            self.model = model
            self._initialized = True
            self._last_error = None
            logger.info("Initialized model %s on %s", self.model_name, self.device)
            return True
        except Exception as exc:
            self._release_resources()
            self._record_failure("Model initialization failed", exc)
            return False

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = 64,
        temperature: float = 0.7,
        num_beams: int = 1,
    ) -> Optional[str]:
        """Generate a continuation for ``prompt``.

        Args:
            prompt: Non-empty input text.
            max_new_tokens: Maximum continuation length. ``None`` uses up to
                :data:`MAX_AUTOMATIC_NEW_TOKENS` within the remaining context.
            temperature: Sampling temperature. Set to ``0`` for deterministic
                decoding.
            num_beams: Beam count. The default of one minimizes latency and
                memory use on constrained systems.

        Returns:
            The decoded continuation, or ``None`` when inference fails.

        Raises:
            ValueError: If generation parameters are invalid.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if max_new_tokens is not None and max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive or None")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if num_beams < 1:
            raise ValueError("num_beams must be at least 1")
        if self._failure_threshold_reached:
            logger.error(
                "Text generation disabled after %d failures", self._max_failures
            )
            return None
        if not self.initialize():
            return None

        with self._generation_semaphore:
            return self._generate_unlocked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_beams=num_beams,
            )

    def _generate_unlocked(
        self,
        *,
        prompt: str,
        max_new_tokens: Optional[int],
        temperature: float,
        num_beams: int,
    ) -> Optional[str]:
        """Generate text while holding the generation lock."""

        torch_module = self._torch
        tokenizer = self.tokenizer
        model = self.model
        if torch_module is None or tokenizer is None or model is None:
            self._record_failure(
                "Text generation failed",
                RuntimeError("Model resources are not initialized"),
            )
            return None

        try:
            formatted_prompt = self._format_prompt(tokenizer, prompt)
            input_tokens = tokenizer(formatted_prompt, return_tensors="pt")
            input_ids = input_tokens.get("input_ids")
            if input_ids is None or not isinstance(input_ids, torch_module.Tensor):
                raise TypeError("Tokenizer did not return a Torch input_ids tensor")
            input_ids = input_ids.to(self.device)

            attention_mask = input_tokens.get("attention_mask")
            if attention_mask is not None:
                if not isinstance(attention_mask, torch_module.Tensor):
                    raise TypeError("Tokenizer attention_mask is not a Torch tensor")
                attention_mask = attention_mask.to(self.device)

            input_length = int(input_ids.shape[-1])
            model_config = getattr(model, "config", None)
            model_limit = int(
                getattr(model_config, "max_position_embeddings", 2048) or 2048
            )
            remaining_context = max(1, model_limit - input_length)
            generation_length = (
                min(remaining_context, MAX_AUTOMATIC_NEW_TOKENS)
                if max_new_tokens is None
                else min(max_new_tokens, remaining_context)
            )

            do_sample = temperature > 0
            generation_kwargs: Dict[str, Any] = {
                "max_new_tokens": generation_length,
                "num_beams": num_beams,
                "do_sample": do_sample,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature

            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if isinstance(pad_token_id, list):
                pad_token_id = pad_token_id[0] if pad_token_id else None
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0] if eos_token_id else None
            if pad_token_id is None:
                pad_token_id = eos_token_id
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id

            inference_context = (
                torch_module.inference_mode()
                if hasattr(torch_module, "inference_mode")
                else torch_module.no_grad()
            )
            with inference_context:
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )

            sequences = (
                output
                if isinstance(output, torch_module.Tensor)
                else getattr(output, "sequences", None)
            )
            if sequences is None or not isinstance(sequences, torch_module.Tensor):
                raise TypeError(
                    f"Model returned unsupported output type {type(output).__name__}"
                )
            first_sequence = sequences[0] if sequences.ndim > 1 else sequences
            generated_ids = first_sequence[input_length:]
            result = tokenizer.decode(generated_ids, skip_special_tokens=True)
            self._last_error = None
            return str(result).strip()
        except Exception as exc:
            self._record_failure("Text generation failed", exc)
            return None

    def query(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = 256,
        temperature: float = 0.7,
        num_beams: int = 1,
    ) -> Optional[str]:
        """Generate a model response using the standard inference path."""

        return self.generate_text(prompt, max_new_tokens, temperature, num_beams)

    def close(self) -> None:
        """Release references to model resources and accelerator caches."""

        for _ in range(self._concurrency_limit):
            self._generation_semaphore.acquire()
        try:
            with self._initialization_lock:
                self._release_resources()
        finally:
            for _ in range(self._concurrency_limit):
                self._generation_semaphore.release()

    @staticmethod
    def _format_prompt(tokenizer: Any, prompt: str) -> str:
        """Apply an instruction model's chat template when one is available."""

        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_template):
            return prompt
        try:
            rendered = apply_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, TypeError, ValueError):
            logger.debug(
                "Tokenizer for %s has no usable chat template; using raw prompt",
                type(tokenizer).__name__,
            )
            return prompt
        return rendered if isinstance(rendered, str) and rendered else prompt

    def _release_resources(self) -> None:
        """Release loaded objects without importing optional dependencies."""

        self.model = None
        self.tokenizer = None
        self._initialized = False
        torch_module = self._torch
        if (
            torch_module is not None
            and hasattr(torch_module, "cuda")
            and torch_module.cuda.is_available()
        ):
            torch_module.cuda.empty_cache()
        self._torch = None
        self.device = (
            self._requested_device
            if self._requested_device is not None
            else "uninitialized"
        )

    def _record_failure(self, message: str, error: Exception) -> None:
        """Record an inference failure and disable repeatedly failing models."""

        self._inference_failures += 1
        self._last_error = f"{message} for '{self.model_name}': {error}"
        if self._inference_failures >= self._max_failures:
            self._failure_threshold_reached = True
        logger.error(self._last_error)

    def __enter__(self) -> ModelState:
        """Return this model state for context-managed use."""

        return self

    def __exit__(self, *_exc: object) -> None:
        """Release model resources when leaving a context."""

        self.close()


__all__ = ["MAX_AUTOMATIC_NEW_TOKENS", "ModelDependencyError", "ModelState"]
