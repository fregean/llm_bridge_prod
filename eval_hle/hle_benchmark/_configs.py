from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Config:
    dataset: str
    provider: str
    base_url: str
    model: str
    max_tokens: int
    reasoning: bool
    num_workers: int
    max_samples: int
    judge: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    question_indices: Optional[List[int]] = None
    question_range: Optional[List[int]] = None