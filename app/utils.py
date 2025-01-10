import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load tokenizer and model
MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def preprocess_text(text: str) -> str:
    """
    Basic text cleaning:
    - Lowercase
    - Remove URLs
    - Remove non-alphanumeric (except basic punctuation)
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z0-9\s.,!?']", "", text)  # Keep basic punctuation
    text = text.strip()  # Strip extra whitespace
    return text

def analyze_transcript(file_path: Path) -> dict:
    """
    Analyze transcript for sentence-level sentiment and aggregate polarity and intensity.
    """
    with file_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # Parse transcript
    pattern = r"\[(.*?)(?:\s\d{2}:\d{2})?\]\s*(.+?)(?=\n\[|\Z)"
    matches = re.findall(pattern, content, flags=re.DOTALL)

    sentence_details = []
    speaker_scores = {}
    aggregated_scores = {}

    for speaker_info, sentence in matches:
        speaker_info = speaker_info.strip()
        sentence = preprocess_text(sentence.strip())
        if not sentence: continue

        # Perform sentiment analysis
        encoded_input = tokenizer(sentence, return_tensors="pt")
        output = model(**encoded_input)
        scores = softmax(output.logits.detach().numpy()[0])

        positive_score = scores[1]  # Positive
        negative_score = scores[0]  # Negative
        polarity = positive_score - negative_score
        intensity = abs(polarity)
        dominant_label = "POSITIVE" if polarity > 0 else "NEGATIVE"

        sentence_details.append({
            "speaker": speaker_info,
            "sentence": sentence,
            "positive_score": positive_score,
            "negative_score": negative_score,
            "polarity": polarity,
            "intensity": intensity,
            "dominant_label": dominant_label,
        })

        # Aggregate scores by speaker
        if speaker_info not in speaker_scores:
            speaker_scores[speaker_info] = {"negative": 0.0, "positive": 0.0}

        speaker_scores[speaker_info]["negative"] += negative_score
        speaker_scores[speaker_info]["positive"] += positive_score

    # Compute overall scores for each speaker
    overall_scores = {}
    for speaker, scores in speaker_scores.items():
        overall_scores[speaker] = {
            "overall_sentiment": "POSITIVE" if scores["positive"] > scores["negative"] else "NEGATIVE",
            "scores": scores
        }

    return {
        "sentence_details": sentence_details,
        "overall_scores": overall_scores,  # Add this field to match the SentimentResponse model
    }