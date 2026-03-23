from dataclasses import dataclass

@dataclass
class Event:
  patient: str
  relation: str
  target: str
  time: str
  source: str