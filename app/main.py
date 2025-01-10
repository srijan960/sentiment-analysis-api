from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from utils import analyze_transcript
from models import SentimentResponse
import shutil
import os

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload/", response_model=SentimentResponse)
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "text/plain":
        raise HTTPException(status_code=400, detail="Only text files are allowed.")
    
    print(f"Received file: {file.filename}")
    # Save file
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Analyze transcript
    results = analyze_transcript(file_path)
    return SentimentResponse(file_name=file.filename, sentiment_results=results)

@app.get("/")
def root():
    return {"message": "Welcome to the Sentiment Analysis API"}