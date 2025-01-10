# FastAPI Sentiment Analysis Backend

This project is a FastAPI-based backend for performing sentiment analysis on conversation transcripts. It utilizes a pre-trained transformer model (`distilbert-base-uncased-finetuned-sst-2-english`) to analyze sentiments at a sentence level and aggregates results for individual speakers.

---

## Features

- **Sentiment Analysis:** Detects positive and negative sentiment for each sentence in a conversation transcript.
- **Speaker-Specific Analysis:** Aggregates sentiment scores for individual speakers.
- **Drastic Sentiment Shifts:** Identifies sentences causing drastic shifts in sentiment.

---

## Requirements

- Python 3.8 or higher
- `pip` for Python package management

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/srijan960/sentiment-analysis-api.git
cd your-repo
```

**2. Create a Virtual Environment**

It is recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

**3. Install Dependencies**

Install the required Python packages listed in the requirements.txt file.

```bash
pip install -r requirements.txt
```

**4. Start the Backend**

Run the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

This will start the backend at http://127.0.0.1:8000.

**API Endpoints**

**1. Upload Transcript for Sentiment Analysis**

**Endpoint:** POST /upload/

**Description:** Upload a text file containing the conversation transcript for sentiment analysis.

**Request Parameters**

• **File**: A .txt file containing the transcript.

• **Optional Query Parameters**:

• target_speaker: Specify a speaker to focus on for sentiment analysis.

• threshold: A threshold value to detect drastic sentiment shifts (default: 0.5).

**Example:**

```bash
curl -X POST "http://127.0.0.1:8000/upload/" \
  -F "file=@example_transcript.txt" \
  -F "target_speaker=Speaker 1" \
  -F "threshold=0.6"
```

**File Structure**

```bash
project-root/
├── main.py                # FastAPI entry point
├── utils.py               # Sentiment analysis logic
├── models.py              # Data models for API responses
├── requirements.txt       # List of dependencies
└── uploads/               # Directory to store uploaded transcripts
```
