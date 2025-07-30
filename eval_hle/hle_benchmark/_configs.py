from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Config:
    dataset: str
    provider: str
    base_url: str
    model: str
    max_tokens: int # Deepseek用の最大トークン数
    reasoning: bool
    num_workers: int
    max_samples: int
    judge: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    question_indices: Optional[List[int]] = None
    question_range: Optional[List[int]] = None
    max_completion_tokens: Optional[int] = None  # rubric評価のOpenAI API用の最大トークン数
    predictions_file: Optional[str] = None  # Custom predictions file path