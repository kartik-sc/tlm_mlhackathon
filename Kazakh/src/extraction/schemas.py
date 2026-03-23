from dataclasses import dataclass

@dataclass
class Event:
    patient: str
    relation: str
    source_entity: str
    target_entity: str
    time: str
    evidence: str