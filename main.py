from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import io
import os
import httpx
import json
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, ValidationError

# -----------------------------
# CONFIG
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "25"))

# -----------------------------
# APP CONFIGURATION
# -----------------------------
app = FastAPI(title="Simple OCR + Gemini API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Pydantic Models
# -----------------------------
class ExpenseLine(BaseModel):
    description: Optional[str]
    amount: Optional[str]

class ReceiptData(BaseModel):
    merchant: Optional[str]
    date: Optional[str]
    amount: Optional[str]
    currency: Optional[str]
    expense_type: Optional[str]
    description: Optional[str]
    expense_lines: Optional[List[ExpenseLine]]

# -----------------------------
# Gemini LLM CALL
# -----------------------------
async def call_gemini_llm(ocr_text: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set.")

    prompt = (
        "You are a strict JSON extractor. Given OCR text from a receipt, return ONLY a single valid JSON object with keys (if available):\n"
        "merchant, date (YYYY-MM-DD preferred), amount, currency, expense_type, description, expense_lines (array of {description, amount}).\n"
        "If a value is missing, set it to null. Do NOT include extra commentary.\n\n"
        f"OCR_TEXT:\n{ocr_text}\n"
    )

    url = f"{GEMINI_ENDPOINT}/{GEMINI_MODEL}:generateContent"

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        response = r.json()

    # extract textual output from Gemini response
    raw_text = None
    if isinstance(response, dict):
        if "candidates" in response and len(response["candidates"]) > 0:
            first = response["candidates"][0]
            raw_text = first.get("content", {}).get("parts", [{}])[0].get("text")
        if not raw_text:
            raw_text = response.get("output") or response.get("text")
        if not raw_text:
            # fallback deep search
            def find_text(obj):
                if isinstance(obj, str):
                    return obj
                if isinstance(obj, dict):
                    for v in obj.values():
                        found = find_text(v)
                        if found:
                            return found
                if isinstance(obj, list):
                    for item in obj:
                        found = find_text(item)
                        if found:
                            return found
                return None
            raw_text = find_text(response)

    if not raw_text:
        raise RuntimeError("Could not extract text from Gemini response.")

    # try to parse JSON from Gemini output
    try:
        parsed_json = json.loads(raw_text)
        if isinstance(parsed_json, dict):
            return parsed_json
    except Exception:
        # fallback: return raw text under 'llm_raw'
        return {"llm_raw": raw_text.strip()}

    return {}

# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/extract-receipt")
async def extract_receipt(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        # read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # simple OCR (no preprocessing)
        text = pytesseract.image_to_string(image, lang="eng")

        # -----------------------------
        # lightweight OCR cleanup
        # -----------------------------
        text_cleaned = text
        text_cleaned = "".join(c if c.isprintable() else " " for c in text_cleaned)
        text_cleaned = text_cleaned.replace("O", "0").replace("l", "1")
        text_cleaned = "\n".join(line.strip() for line in text_cleaned.splitlines() if line.strip())

        structured: Dict[str, Any] = {}
        llm_used = False

        if GEMINI_API_KEY:
            try:
                llm_used = True
                structured = await call_gemini_llm(text_cleaned)
            except Exception as e:
                structured = {"llm_raw": text_cleaned.strip(), "_llm_error": str(e)}
        else:
            structured = {"llm_raw": text_cleaned.strip()}

        # validate via pydantic
        try:
            validated = ReceiptData(**structured)
            structured = validated.dict()
        except ValidationError as ve:
            structured["_validation_error"] = str(ve)

        return {
            "filename": file.filename,
            "llm_used": llm_used,
            "extracted_text": text_cleaned.strip(),
            "structured_data": structured
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Simple OCR API is running ✅"}
