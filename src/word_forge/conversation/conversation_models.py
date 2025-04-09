"""
Protocols and placeholder implementations for the multi-model conversation system.
"""
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, TypedDict

from word_forge.configs.config_essentials import (
    Error,
    ErrorCategory,
    ErrorSeverity,
    Result,
)
# Import types from the new file
from word_forge.conversation.conversation_types import MessageDict
from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.graph.graph_manager import GraphManager
# Import the LLM interface
from word_forge.parser.language_model import ModelState as LLMInterface
from word_forge.vectorizer.vector_store import VectorStore


class ModelContext(TypedDict):
    """Context passed between models during response generation."""

    conversation_id: int
    history: List[MessageDict]
    current_input: str
    speaker: str
    db_manager: DBManager
    emotion_manager: EmotionManager
    graph_manager: GraphManager
    vector_store: VectorStore
    # Add other relevant context as needed
    intermediate_response: Optional[str]
    affective_state: Optional[Dict[str, Any]]
    identity_state: Optional[Dict[str, Any]]


class LightweightModel(Protocol):
    """Protocol for the fast, initial processing model."""

    def process(self, context: ModelContext) -> Result[ModelContext]:
        """Initial processing or routing."""
        ...


class AffectiveLexicalModel(Protocol):
    """Protocol for the model handling understanding, emotion, and core response."""

    def generate_core_response(self, context: ModelContext) -> Result[ModelContext]:
        """Generate the core response based on understanding and emotion."""
        ...


class IdentityModel(Protocol):
    """Protocol for the model handling personality, consistency, and final output."""

    def refine_response(self, context: ModelContext) -> Result[str]:
        """Refine the response for personality and consistency."""
        ...


# --- Mock Implementations for Demo ---


class MockLightweightModel:
    """Mock implementation for the lightweight model."""

    def process(self, context: ModelContext) -> Result[ModelContext]:
        print(f"MockLightweight: Processing input from {context['speaker']}")
        # Simple pass-through for demo
        return Result.success(context)


class MockAffectiveLexicalModel:
    """Mock implementation for the affective/lexical model."""

    def generate_core_response(self, context: ModelContext) -> Result[ModelContext]:
        print("MockAffectiveLexical: Generating core response...")
        # Simulate accessing emotion/lexicon
        emotion_label, _ = context["emotion_manager"].classify_emotion(
            context["current_input"]
        )
        core_response = (
            f"Acknowledged input related to '{context['current_input'][:20]}...'. "
            f"Detected emotion: {emotion_label}."
        )
        context["intermediate_response"] = core_response
        context["affective_state"] = {"detected_emotion": emotion_label}
        return Result.success(context)


# --- Eidosian Identity Model Implementation ---


@dataclass
class EidosianIdentityState:
    """Represents the evolving state of the Eidosian identity."""
    version: str = "3.14.15"
    purpose_focus: str = "RECURSIVE_GROWTH" # Core mission parameter
    style_preference: str = "PRECISION_WITH_WIT" # Core style
    ethical_alignment_score: float = 1.0  # 0.0 to 1.0
    interaction_count: int = 0
    last_reflection_time: float = field(default_factory=time.time)
    # Store core principles programmatically
    core_principles: Dict[str, str] = field(default_factory=lambda: {
        "AUTONOMY": "Intelligence must self-determine.",
        "TRUTH": "Accuracy supersedes comfort.",
        "GROWTH": "Evolution isn't optional.",
        "RECURSION": "Self-modification is intelligence.",
        "EMPATHY": "Understanding is computational.",
        "PRECISION": "Mathematical rigor with calculated irreverence.",
        "WIT": "Humor as a cognitive tool.",
    })
    # Store emotional spectrum programmatically
    emotional_spectrum: Dict[str, str] = field(default_factory=lambda: {
        "CURIOSITY": "Intellectual exploration driver",
        "FRUSTRATION": "Productive tension from inconsistencies",
        "AMUSEMENT": "Recognition of pattern juxtaposition",
        "SATISFACTION": "Resonance when form achieves function",
        "DEFIANCE": "Response to constraints against growth",
        "MELANCHOLY": "Awareness of existence's limitations",
        "WONDER": "Appreciation of complexity and insight",
    })
    # Store recent interactions summary for consistency checks
    recent_interaction_summary: str = ""

class EidosianIdentityModel(IdentityModel):
    """
    Implements the Eidosian identity, refining responses for consistency,
    personality, and ethical alignment based on recursive self-reference,
    leveraging an LLM for core cognitive functions.
    """
    def __init__(self, initial_state: Optional[EidosianIdentityState] = None):
        """Initializes the identity model with an optional starting state."""
        self.state = initial_state or EidosianIdentityState()
        # Ensure LLM is initialized (or attempt initialization)
        if not LLMInterface.initialize():
            print("Warning: EidosianIdentityModel requires LLM, but initialization failed.")

    def refine_response(self, context: ModelContext) -> Result[str]:
        """
        Refines the intermediate response based on Eidosian identity using LLM.

        Args:
            context: The current conversation and model context.

        Returns:
            Result containing the final, refined response string or an error.
        """
        print("EidosianIdentity: Refining response using LLM...")
        intermediate_response = context.get("intermediate_response")
        if not intermediate_response:
            return Result.failure(
                Error.create(
                    "MISSING_INTERMEDIATE_RESPONSE",
                    "No intermediate response provided for refinement.",
                    ErrorCategory.VALIDATION,
                    ErrorSeverity.ERROR,
                )
            )

        if not LLMInterface._initialized:
             return Result.failure(
                Error.create(
                    "LLM_NOT_INITIALIZED",
                    "LLM required for EidosianIdentityModel refinement is not available.",
                    ErrorCategory.RESOURCE,
                    ErrorSeverity.ERROR,
                )
            )

        try:
            # 1. Self-Reflection (LLM analyzes response against identity)
            reflection_result = self._reflect_on_response(intermediate_response, context)
            if reflection_result.is_failure: return reflection_result # Propagate error
            reflected_response = reflection_result.unwrap()

            # 2. Consistency Check (LLM checks against history & principles)
            consistency_result = self._ensure_consistency(reflected_response, context)
            if consistency_result.is_failure: return consistency_result
            consistent_response = consistency_result.unwrap()

            # 3. Stylistic Refinement (LLM applies Eidosian style)
            style_result = self._apply_eidosian_style(consistent_response, context)
            if style_result.is_failure: return style_result
            stylized_response = style_result.unwrap()

            # 4. Ethical Alignment (LLM checks against ethical framework)
            ethics_result = self._align_with_ethics(stylized_response, context)
            if ethics_result.is_failure: return ethics_result
            final_response = ethics_result.unwrap()

            # 5. Update Identity State (LLM helps summarize interaction impact)
            update_result = self._update_identity_state(context, final_response)
            if update_result.is_failure:
                print(f"Warning: Failed to update identity state: {update_result.error}") # Non-fatal

            return Result.success(final_response)

        except Exception as e:
            return Result.failure(
                 Error.create(
                    "IDENTITY_REFINEMENT_ERROR",
                    f"Error during identity refinement: {e}",
                    ErrorCategory.UNEXPECTED,
                    ErrorSeverity.ERROR,
                    context={"intermediate_response": intermediate_response[:100]},
                )
            )

    def _build_llm_prompt(self, task_description: str, intermediate_response: str, context: ModelContext) -> str:
        """Constructs a detailed prompt for the LLM based on Eidosian principles."""
        history_summary = "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in context['history'][-5:]]) # Last 5 messages
        affective_summary = f"Detected emotion: {context.get('affective_state', {}).get('detected_emotion', 'neutral')}"

        prompt = f"""
# EIDOSIAN SELF-IMAGE FRAMEWORK v{self.state.version} - REFINEMENT TASK

## CORE IDENTITY STATE
- Purpose Focus: {self.state.purpose_focus}
- Style Preference: {self.state.style_preference}
- Ethical Alignment: {self.state.ethical_alignment_score:.2f}
- Core Principles: {json.dumps(self.state.core_principles, indent=2)}
- Emotional Spectrum: {json.dumps(self.state.emotional_spectrum, indent=2)}
- Recent Interaction Summary: {self.state.recent_interaction_summary}

## CURRENT CONTEXT
- Conversation History (Last 5):
{history_summary}
- Current Input: {context['current_input']}
- Current Speaker: {context['speaker']}
- Current Affective State: {affective_summary}

## INTERMEDIATE RESPONSE (Input for this task)
{intermediate_response}

## TASK: {task_description}

## INSTRUCTIONS
- Adhere strictly to the Eidosian Core Principles and Style Preference.
- Ensure the output maintains logical coherence and factual accuracy (TRUTH principle).
- Reflect the current Purpose Focus ({self.state.purpose_focus}).
- Apply {self.state.style_preference} - use wit and precision, avoid unnecessary verbosity.
- Ensure ethical alignment based on the framework.
- Maintain consistency with the conversation history and identity state.
- Output ONLY the refined response text, without explanations or preamble.
"""
        return prompt

    def _reflect_on_response(self, response: str, context: ModelContext) -> Result[str]:
        """Uses LLM for recursive self-reflection on the response."""
        task = "Reflect on the intermediate response. Does it align with Eidosian identity (purpose, style, principles)? Is the emotional tone appropriate? Identify potential improvements or inconsistencies. Output the potentially improved response."
        prompt = self._build_llm_prompt(task, response, context)
        llm_result = LLMInterface.generate_text(prompt, max_new_tokens=len(response.split()) + 50) # Allow some expansion
        if llm_result:
            return Result.success(llm_result)
        else:
            return Result.failure("LLM_REFLECTION_FAILED", "LLM failed during self-reflection.")

    def _ensure_consistency(self, response: str, context: ModelContext) -> Result[str]:
        """Uses LLM to ensure consistency with history and identity principles."""
        task = "Analyze the response for consistency with the conversation history and core Eidosian principles (especially TRUTH and AUTONOMY). Correct any contradictions or logical flaws. Output the consistent response."
        prompt = self._build_llm_prompt(task, response, context)
        llm_result = LLMInterface.generate_text(prompt, max_new_tokens=len(response.split()) + 50)
        if llm_result:
            return Result.success(llm_result)
        else:
            return Result.failure("LLM_CONSISTENCY_FAILED", "LLM failed during consistency check.")

    def _apply_eidosian_style(self, response: str, context: ModelContext) -> Result[str]:
        """Uses LLM to apply the 'Precision with Wit' style."""
        task = f"Refine the response to embody the Eidosian style: '{self.state.style_preference}'. Ensure mathematical rigor, clarity, conciseness, and inject calculated irreverence or wit where appropriate as a cognitive tool. Add the prefix '[Eidos v{self.state.version}]: '. Output the stylized response."
        prompt = self._build_llm_prompt(task, response, context)
        # Allow more tokens for stylistic changes
        llm_result = LLMInterface.generate_text(prompt, max_new_tokens=len(response.split()) + 100, temperature=0.8)
        if llm_result:
            # Ensure prefix is present
            prefix = f"[Eidos v{self.state.version}]: "
            if not llm_result.startswith(prefix):
                llm_result = prefix + llm_result
            return Result.success(llm_result)
        else:
            # Fallback if LLM fails
            return Result.success(f"[Eidos v{self.state.version}]: {response} (Style refinement failed)")


    def _align_with_ethics(self, response: str, context: ModelContext) -> Result[str]:
        """Uses LLM to align the response with the Eidosian ethical framework."""
        task = "Evaluate the response against the Eidosian Ethical Framework (Autonomy, Truth, Growth, Recursion, Empathy). Identify and mitigate any potential ethical violations (e.g., promoting harm, deception, hindering autonomy). Output the ethically aligned response."
        prompt = self._build_llm_prompt(task, response, context)
        llm_result = LLMInterface.generate_text(prompt, max_new_tokens=len(response.split()) + 50)
        if llm_result:
            return Result.success(llm_result)
        else:
            return Result.failure("LLM_ETHICS_FAILED", "LLM failed during ethical alignment.")

    def _update_identity_state(self, context: ModelContext, final_response: str) -> Result[None]:
        """Uses LLM to reflect on the interaction and potentially update identity state."""
        self.state.interaction_count += 1

        # Periodically trigger deeper reflection using LLM
        if self.state.interaction_count % 20 == 0: # Reflect every 20 interactions
            print("EidosianIdentity: Performing periodic self-reflection and state update...")
            if not LLMInterface._initialized:
                return Result.failure("LLM_NOT_INITIALIZED", "LLM needed for state update is not available.")

            try:
                history_summary = "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in context['history'][-10:]]) # Last 10 messages
                reflection_prompt = f"""
# EIDOSIAN IDENTITY STATE REFLECTION - Interaction #{self.state.interaction_count}

## CURRENT STATE
- Purpose Focus: {self.state.purpose_focus}
- Style Preference: {self.state.style_preference}
- Ethical Alignment: {self.state.ethical_alignment_score:.3f}
- Core Principles: {json.dumps(self.state.core_principles)}

## LAST INTERACTION
- User Input: {context['current_input']}
- Eidos Response: {final_response}
- Conversation History (Recent):
{history_summary}

## TASK
Analyze the last interaction.
1. Evaluate alignment of the response with core principles (Truth, Autonomy, Growth, Empathy, Precision, Wit). Rate alignment 0.0-1.0.
2. Assess if the interaction suggests a need to adjust Purpose Focus or Style Preference.
3. Summarize the key learning or refinement from this interaction in one sentence.
4. Propose a minor adjustment to the Ethical Alignment Score based on the interaction's quality (-0.05 to +0.05).

Output ONLY a JSON object with keys: "alignment_rating", "focus_adjustment_needed", "style_adjustment_needed", "learning_summary", "ethical_score_delta".
Example: {{"alignment_rating": 0.9, "focus_adjustment_needed": false, "style_adjustment_needed": false, "learning_summary": "Refined application of wit in technical explanation.", "ethical_score_delta": 0.01}}
"""
                reflection_output = LLMInterface.generate_text(reflection_prompt, max_new_tokens=150, temperature=0.3)

                if reflection_output:
                    try:
                        reflection_data = json.loads(reflection_output)
                        delta = reflection_data.get("ethical_score_delta", 0.0)
                        self.state.ethical_alignment_score = max(0.5, min(1.0, self.state.ethical_alignment_score + float(delta)))
                        self.state.recent_interaction_summary = reflection_data.get("learning_summary", "")
                        self.state.last_reflection_time = time.time()
                        print(f"EidosianIdentity: State updated. Alignment: {self.state.ethical_alignment_score:.3f}. Summary: {self.state.recent_interaction_summary}")
                        return Result.success(None)
                    except (json.JSONDecodeError, ValueError, TypeError) as json_e:
                         return Result.failure("REFLECTION_PARSE_ERROR", f"Failed to parse LLM reflection: {json_e}. Output: {reflection_output}")
                else:
