"""
ChatML Format Converter
=======================

WHAT THIS MODULE DOES:
Converts clinical notes into ChatML format (Chat Markup Language), which is the
native format expected by Phi-3 and other instruction-tuned models.

WHY WE NEED THIS:
1. **Model Compatibility**: Phi-3 was fine-tuned using ChatML format during instruction
   tuning. Using the same format during our fine-tuning ensures the model recognizes
   the structure and learns effectively.

2. **Structured Outputs**: By formatting the assistant's response as JSON, we make
   parsing deterministic. The model learns to output well-formed JSON instead of
   free-text that requires complex regex parsing.

3. **Conversation Format**: ChatML uses a conversation structure (system/user/assistant)
   which is more natural for instruction-following tasks than raw text completion.

HOW IT WORKS:
1. Take a clinical note and its ICD-10 codes
2. Create a 3-message conversation:
   - System: Defines the assistant's role
   - User: Provides the clinical note
   - Assistant: Responds with JSON-formatted codes
3. The model learns to predict the assistant message given the system+user context

EDUCATIONAL NOTE - Fine-Tuning Format:
During fine-tuning, only the ASSISTANT message is used for loss calculation.
The system and user messages provide context, but we don't train the model to
predict them. This is called "causal language modeling with masking."
"""

import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from icd10_fine_tune.data.loader import ClinicalNote


# ============================================================================
# CHAT MESSAGE MODELS
# ============================================================================

class ChatMessage(BaseModel):
    """
    A single message in a chat conversation.
    
    EDUCATIONAL NOTE - ChatML Structure:
    ChatML is a simple but effective format:
    - role: Identifies who is speaking ('system', 'user', 'assistant')
    - content: The actual text content
    
    This structure allows models to understand multi-turn conversations
    and respond appropriately based on context.
    """
    
    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content (text)")


class ChatMLSample(BaseModel):
    """
    A complete training sample in ChatML format.
    
    This represents one complete fine-tuning example. During training,
    the model sees the full conversation but only the assistant's response
    contributes to the loss function.
    """
    
    messages: List[ChatMessage] = Field(..., description="List of chat messages (system, user, assistant)")
    
    note_id: str = Field(..., description="Original note ID (for tracking)")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================
# The system prompt defines the assistant's behavior. This is crucial for
# fine-tuning quality.
#
# EDUCATIONAL NOTE - Crafting System Prompts:
# A good system prompt for fine-tuning should:
# 1. Be concise but clear
# 2. Define the output format explicitly
# 3. Set appropriate behavior (e.g., "be accurate", "use medical knowledge")
# 4. Match the style used during the model's instruction tuning
#
# Avoid:
# - Overly long prompts (wastes tokens)
# - Vague instructions ("help the user" - help how?)
# - Conflicting instructions

DEFAULT_SYSTEM_PROMPT = """You are a medical coding assistant. Analyze clinical notes and assign accurate ICD-10 codes. Always respond with a JSON object containing an array of ICD-10 codes."""

# Alternative system prompts for experimentation
DETAILED_SYSTEM_PROMPT = """You are an expert medical coder. Your task is to carefully analyze clinical notes and assign appropriate ICD-10 diagnosis codes. Consider all diagnoses, symptoms, and conditions mentioned in the note. Respond with a JSON object in the format: {"icd10_codes": ["CODE1", "CODE2", ...]}"""

CONCISE_SYSTEM_PROMPT = """Medical coding assistant. Output: {"icd10_codes": ["CODE1", "CODE2"]}"""


# ============================================================================
# FORMATTER FUNCTIONS
# ============================================================================

def clinical_note_to_chatml(
    note: ClinicalNote,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    include_context: bool = True
) -> ChatMLSample:
    """
    Convert a clinical note into ChatML format for fine-tuning.
    
    Args:
        note: ClinicalNote object with text and codes
        system_prompt: System message defining assistant behavior
        include_context: Whether to include metadata in system prompt
    
    Returns:
        ChatMLSample ready for fine-tuning
    
    EDUCATIONAL NOTE - Context vs Token Budget:
    Including context (patient demographics, encounter type) can improve
    accuracy but uses more tokens. On 6GB VRAM with max_seq_length=512,
    we need to balance informativeness with length.
    
    Rule of thumb:
    - System prompt: ~50 tokens
    - User message: ~200-400 tokens (clinical note)
    - Assistant message: ~20-50 tokens (JSON codes)
    - Total: ~300-500 tokens (fits in 512 limit with margin)
    """
    
    # Build user message with optional context
    user_content = f"Analyze this clinical note:\n\n{note.clinical_text}"
    
    if include_context and note.metadata:
        # Add relevant context if available
        context_parts = []
        if "age" in note.metadata:
            context_parts.append(f"Patient age: {note.metadata['age']}")
        if "gender" in note.metadata:
            context_parts.append(f"Gender: {note.metadata['gender']}")
        if "encounter_type" in note.metadata:
            context_parts.append(f"Encounter: {note.metadata['encounter_type']}")
        
        if context_parts:
            context = "\n".join(context_parts)
            user_content = f"{context}\n\n{user_content}"
    
    # Format assistant response as JSON
    # EDUCATIONAL NOTE - Why JSON Output?
    # 1. Deterministic parsing (no regex needed)
    # 2. Easy to validate during data preprocessing
    # 3. Model learns structured output (useful skill for production)
    # 4. Handles multi-code predictions cleanly
    assistant_content = json.dumps({
        "icd10_codes": note.icd10_codes
    }, ensure_ascii=False)  # ensure_ascii=False handles international characters
    
    # Construct ChatML conversation
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_content),
        ChatMessage(role="assistant", content=assistant_content)
    ]
    
    return ChatMLSample(
        messages=messages,
        note_id=note.note_id,
        metadata={
            "profile": note.profile,
            "num_codes": len(note.icd10_codes),
            **note.metadata
        }
    )


def batch_convert_to_chatml(
    notes: List[ClinicalNote],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    include_context: bool = True
) -> List[ChatMLSample]:
    """
    Convert a batch of clinical notes to ChatML format.
    
    Args:
        notes: List of ClinicalNote objects
        system_prompt: System prompt to use for all samples
        include_context: Whether to include metadata context
    
    Returns:
        List of ChatMLSample objects ready for fine-tuning
    
    EDUCATIONAL NOTE - Batch Processing:
    We process notes in a batch to enable progress tracking and
    error handling. If one note fails to convert, we can log it
    and continue with the rest.
    """
    samples = []
    errors = []
    
    for note in notes:
        try:
            sample = clinical_note_to_chatml(
                note,
                system_prompt=system_prompt,
                include_context=include_context
            )
            samples.append(sample)
        except Exception as e:
            errors.append(f"Note {note.note_id}: {str(e)}")
    
    if errors:
        print(f"WARNING: Failed to convert {len(errors)} notes:")
        for error in errors[:5]:  # Show first 5
            print(f"  {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    return samples


def chatml_to_training_format(sample: ChatMLSample) -> Dict[str, Any]:
    """
    Convert ChatML sample to the exact format expected by HuggingFace trainers.
    
    Args:
        sample: ChatMLSample object
    
    Returns:
        Dictionary with 'messages' key in HF format
    
    EDUCATIONAL NOTE - Training Format:
    HuggingFace's SFTTrainer expects data in this specific format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    
    The trainer will automatically:
    1. Apply the model's chat template (ChatML for Phi-3)
    2. Tokenize the conversation
    3. Mask the non-assistant parts (for loss calculation)
    4. Create attention masks and labels
    """
    return {
        "messages": [
            {"role": msg.role, "content": msg.content}
            for msg in sample.messages
        ]
    }


def save_chatml_dataset(
    samples: List[ChatMLSample],
    output_path: str
) -> None:
    """
    Save ChatML samples to JSONL file (one sample per line).
    
    Args:
        samples: List of ChatMLSample objects
        output_path: Path to output JSONL file
    
    EDUCATIONAL NOTE - Why JSONL?
    JSONL (JSON Lines) is preferred over JSON for training data because:
    1. Easy to stream (load one sample at a time)
    2. Fault tolerant (one bad line doesn't corrupt entire file)
    3. Easy to split/combine (cat, head, tail work)
    4. Standard format for HuggingFace datasets
    """
    from pathlib import Path
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            # Convert to training format and write as JSON line
            training_dict = chatml_to_training_format(sample)
            # Add metadata for tracking
            training_dict["note_id"] = sample.note_id
            training_dict["metadata"] = sample.metadata
            
            json.dump(training_dict, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"Saved {len(samples)} samples to {output_path}")
