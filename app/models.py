from pydantic import BaseModel
from typing import Dict, List


class SentenceSentiment(BaseModel):
    speaker: str
    sentence: str
    negative_score: float
    positive_score: float
    polarity: float
    intensity: float
    dominant_label: str


class SpeakerSentiment(BaseModel):
    overall_sentiment: str
    scores: Dict[str, float]


class SentimentResults(BaseModel):
    sentence_details: List[SentenceSentiment]
    overall_scores: Dict[str, SpeakerSentiment]


class SentimentResponse(BaseModel):
    file_name: str
    sentiment_results: SentimentResults