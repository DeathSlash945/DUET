"""
Shared data contracts for DUET.

This file defines the standard data schemas used across data preprocessing,
policy fine-tuning, critic training, steering activation collection, and evaluation.
All schemas are dataclasses with dictionary serialization (to_dict/from_dict) and strict type validation.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal


DatasetVersion = Literal["original", "revised"]


@dataclass
class ProcessedRecord:
    """
    Standard processed dialogue record produced by the PERSONA-CHAT data pipeline.
    
    Attributes:
        id: Unique identifier for the dialogue turn (e.g. "train_000123").
        history: List of prior dialogue turns leading up to the target.
        profile_self: List of persona sentences for the speaker (self).
        profile_partner: List of persona sentences for the dialogue partner.
        target: The true (gold) next response string.
        persona_labels: List of label indices or IDs indicating which profile 
                        attribute(s) grounded the target response.
        dataset_version: Specifies whether this record is from the "original" 
                         or "revised" PERSONA-CHAT dataset.
    """
    id: str
    history: List[str]
    profile_self: List[str]
    profile_partner: List[str]
    target: str
    persona_labels: List[int]
    dataset_version: DatasetVersion

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Field 'id' must be a non-empty string.")
        
        if not isinstance(self.history, list) or not all(isinstance(x, str) for x in self.history):
            raise ValueError("Field 'history' must be a list of strings.")
            
        if not isinstance(self.profile_self, list) or not all(isinstance(x, str) for x in self.profile_self):
            raise ValueError("Field 'profile_self' must be a list of strings.")
            
        if not isinstance(self.profile_partner, list) or not all(isinstance(x, str) for x in self.profile_partner):
            raise ValueError("Field 'profile_partner' must be a list of strings.")
            
        if not isinstance(self.target, str):
            raise ValueError("Field 'target' must be a string.")
            
        if not isinstance(self.persona_labels, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in self.persona_labels):
            raise ValueError("Field 'persona_labels' must be a list of integers.")
            
        if self.dataset_version not in ("original", "revised"):
            raise ValueError("Field 'dataset_version' must be either 'original' or 'revised'.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a standard dictionary format for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedRecord":
        """Create a ProcessedRecord instance from a standard dictionary."""
        if not isinstance(data, dict):
            raise TypeError("Input data must be a dictionary.")
            
        required_fields = ["id", "history", "profile_self", "profile_partner", "target", "persona_labels", "dataset_version"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise KeyError(f"Missing required field(s) in ProcessedRecord: {missing}")

        return cls(
            id=data["id"],
            history=list(data["history"]) if isinstance(data["history"], (list, tuple)) else data["history"],
            profile_self=list(data["profile_self"]) if isinstance(data["profile_self"], (list, tuple)) else data["profile_self"],
            profile_partner=list(data["profile_partner"]) if isinstance(data["profile_partner"], (list, tuple)) else data["profile_partner"],
            target=data["target"],
            persona_labels=list(data["persona_labels"]) if isinstance(data["persona_labels"], (list, tuple)) else data["persona_labels"],
            dataset_version=data["dataset_version"],
        )


@dataclass
class CriticCandidateSet:
    """
    Candidate set constructed for persona critic identification training & evaluation.
    
    Attributes:
        id: Unique identifier for the sample.
        response: Candidate response text being evaluated by the critic.
        true_profile_id: Identifier or index of the correct ground-truth profile.
        distractor_profiles: List of distractor persona profile lists (each a list of strings).
        is_copy_probe: Flag indicating whether this sample is an adversarial copy-probe negative.
    """
    id: str
    response: str
    true_profile_id: str
    distractor_profiles: List[List[str]]
    is_copy_probe: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Field 'id' must be a non-empty string.")
        if not isinstance(self.response, str):
            raise ValueError("Field 'response' must be a string.")
        if not isinstance(self.true_profile_id, str):
            raise ValueError("Field 'true_profile_id' must be a string.")
        
        if not isinstance(self.distractor_profiles, list):
            raise ValueError("Field 'distractor_profiles' must be a list of profiles.")
        for p in self.distractor_profiles:
            if not isinstance(p, list) or not all(isinstance(x, str) for x in p):
                raise ValueError("Each item in 'distractor_profiles' must be a list of strings.")
                
        if not isinstance(self.is_copy_probe, bool):
            raise ValueError("Field 'is_copy_probe' must be strictly a boolean (True or False).")

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate set to dictionary format for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CriticCandidateSet":
        """Create a CriticCandidateSet instance from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError("Input data must be a dictionary.")

        required_fields = ["id", "response", "true_profile_id", "distractor_profiles"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise KeyError(f"Missing required field(s) in CriticCandidateSet: {missing}")

        is_copy_probe = data.get("is_copy_probe", False)
        if not isinstance(is_copy_probe, bool):
            raise ValueError("Field 'is_copy_probe' must be strictly a boolean.")

        distractors = data["distractor_profiles"]
        if isinstance(distractors, (list, tuple)):
            distractors = [list(p) if isinstance(p, (list, tuple)) else p for p in distractors]

        return cls(
            id=data["id"],
            response=data["response"],
            true_profile_id=data["true_profile_id"],
            distractor_profiles=distractors,
            is_copy_probe=is_copy_probe,
        )


@dataclass
class AttributeActivationPair:
    """
    Metadata contract for paired activation collection in Component B (steering).
    
    Attributes:
        attribute_id: Identifier or description of the specific persona attribute (e.g., "likes_dogs").
        expressed_context: Dialogue prompt context where the attribute is expressed.
        matched_context: Paired dialogue prompt context where the attribute is NOT expressed.
        layer_index: Layer index where hidden activations were extracted.
    """
    attribute_id: str
    expressed_context: str
    matched_context: str
    layer_index: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.attribute_id, str) or not self.attribute_id.strip():
            raise ValueError("Field 'attribute_id' must be a non-empty string.")
        if not isinstance(self.expressed_context, str):
            raise ValueError("Field 'expressed_context' must be a string.")
        if not isinstance(self.matched_context, str):
            raise ValueError("Field 'matched_context' must be a string.")
        if self.layer_index is not None and (not isinstance(self.layer_index, int) or isinstance(self.layer_index, bool)):
            raise ValueError("Field 'layer_index' must be an integer if provided.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert attribute activation pair metadata to a standard dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttributeActivationPair":
        """Create an AttributeActivationPair instance from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError("Input data must be a dictionary.")

        required_fields = ["attribute_id", "expressed_context", "matched_context"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise KeyError(f"Missing required field(s) in AttributeActivationPair: {missing}")

        return cls(
            attribute_id=data["attribute_id"],
            expressed_context=data["expressed_context"],
            matched_context=data["matched_context"],
            layer_index=data.get("layer_index"),
        )


@dataclass
class GenerationDump:
    """
    Standard generation dump schema handed off to Sumagna's evaluation harness.
    
    Attributes:
        run_id: Experiment or run identifier.
        model: Name/checkpoint of the policy model evaluated.
        condition: Experimental condition ("sft", "dpo_entailment", "dpo_critic", "steer", "bidir").
        dialog_id: Dialogue ID corresponding to the input sample.
        profile_self: Speaker's persona sentences.
        profile_partner: Partner's persona sentences.
        response: Model generated response text.
    """
    run_id: str
    model: str
    condition: Literal["sft", "dpo_entailment", "dpo_critic", "steer", "bidir"]
    dialog_id: str
    profile_self: List[str]
    profile_partner: List[str]
    response: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("Field 'run_id' must be a non-empty string.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Field 'model' must be a non-empty string.")
        if self.condition not in ("sft", "dpo_entailment", "dpo_critic", "steer", "bidir"):
            raise ValueError("Field 'condition' must be one of: 'sft', 'dpo_entailment', 'dpo_critic', 'steer', 'bidir'.")
        if not isinstance(self.dialog_id, str) or not self.dialog_id.strip():
            raise ValueError("Field 'dialog_id' must be a non-empty string.")
        if not isinstance(self.profile_self, list) or not all(isinstance(x, str) for x in self.profile_self):
            raise ValueError("Field 'profile_self' must be a list of strings.")
        if not isinstance(self.profile_partner, list) or not all(isinstance(x, str) for x in self.profile_partner):
            raise ValueError("Field 'profile_partner' must be a list of strings.")
        if not isinstance(self.response, str):
            raise ValueError("Field 'response' must be a string.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert generation output to standard dictionary for JSONL saving."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationDump":
        """Create a GenerationDump instance from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError("Input data must be a dictionary.")

        required_fields = ["run_id", "model", "condition", "dialog_id", "profile_self", "profile_partner", "response"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise KeyError(f"Missing required field(s) in GenerationDump: {missing}")

        return cls(
            run_id=data["run_id"],
            model=data["model"],
            condition=data["condition"],
            dialog_id=data["dialog_id"],
            profile_self=list(data["profile_self"]) if isinstance(data["profile_self"], (list, tuple)) else data["profile_self"],
            profile_partner=list(data["profile_partner"]) if isinstance(data["profile_partner"], (list, tuple)) else data["profile_partner"],
            response=data["response"],
        )
