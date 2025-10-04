from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import io
import re
import numpy as np
import cv2
import os
import json
import httpx
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ValidationError

# -----------------------------
#  CONFIG
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # <-- set this in your environment (do NOT hardcode)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "25"))

# -----------------------------
#  APP CONFIGURATION
# -----------------------------
app = FastAPI(
    title="Enhanced Tesseract OCR API + Gemini",
    description="Extracts text and structured data (amount, date, merchant, expense lines, expense type) from receipts using Tesseract OCR + OpenCV preprocessing + Google Gemini.",
    version="2.0.0-gemini"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
#  IMAGE PREPROCESSING
# -----------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = np.array(image)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    if max(h, w) < 800:
        scale = 800.0 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return processed


# -----------------------------
#  FALLBACK REGEX PARSING
# -----------------------------

def parse_receipt_text_regex(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    amount_match = re.search(r'([\$₹€])\s?(\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{1,2})?)', text)
    if amount_match:
        result['amount'] = amount_match.group(0)
    else:
        fallback_amount = re.search(r'\b\d{1,6}[\.,]\d{2}\b', text)
        if fallback_amount:
            result['amount'] = fallback_amount.group()

    date_match = re.search(r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b', text)
    if not date_match:
        date_match = re.search(r'\b(\d{4}[/-]\d{2}[/-]\d{2})\b', text)
    if date_match:
        result['date'] = date_match.group(1)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        possible_merchant = lines[0]
        for line in lines[:6]:
            if re.search(r'(RESTAURANT|STORE|HOTEL|CAFE|MART|SHOP|COMPANY|INN|BAR|DINER)', line.upper()):
                possible_merchant = line
                break
        result['merchant'] = possible_merchant

    lines_items: List[Dict[str, str]] = []
    for line in lines:
        m = re.search(r'(.+?)\s+([\d\.,]+)\s*$', line)
        if m and len(m.group(1).strip()) > 2:
            lines_items.append({"description": m.group(1).strip(), "amount": m.group(2).replace(',', '.')})
    if lines_items:
        result['expense_lines'] = lines_items

    return result


# -----------------------------
#  GEMINI LLM CALL
# -----------------------------

async def call_gemini_llm(ocr_text: str) -> Dict[str, Any]:
    """Call Google Gemini REST endpoint (generateContent). Returns parsed JSON (best-effort).

    Uses X-goog-api-key header (API key method). Make sure GEMINI_API_KEY is set.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment. Set it before calling the LLM.")

    prompt = (
        "You are a strict JSON extractor. Given OCR text from a receipt, return ONLY a single valid JSON object with keys (if available):\n"
        "merchant, date (YYYY-MM-DD preferred), amount, currency, expense_type, description, expense_lines (array of {description, amount}).\n"
        "If a value is missing, set it to null. Do NOT include extra commentary.\n\n"
        f"OCR_TEXT:\n{ocr_text}\n"
    )

    url = f"{GEMINI_ENDPOINT}/{GEMINI_MODEL}:generateContent"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        j = r.json()

    # Robust extraction of returned text from common Gemini response shapes
    raw_text: Optional[str] = None
    # Try several common keys
    if isinstance(j, dict):
        # candidate style
        if "candidates" in j and isinstance(j.get("candidates"), list) and len(j["candidates"])>0:
            first = j["candidates"][0]
            # content.parts[0].text (observed structure)
            raw_text = (
                first.get("content", {})
                .get("parts", [{}])[0]
                .get("text") if isinstance(first.get("content"), dict) else None
            )
        # some responses include 'output' or 'text'
        if not raw_text:
            raw_text = j.get("output") or j.get("text")
        # deep search
        if not raw_text:
            try:
                # look for any string values nested inside
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
                raw_text = find_text(j)
            except Exception:
                raw_text = None

    if not raw_text:
        raise RuntimeError("Could not extract textual output from Gemini response")

    # Try to extract JSON object from the text
    json_match = re.search(r"\{[\s\S]*\}", raw_text)
    if json_match:
        json_text = json_match.group(0)
        try:
            parsed = json.loads(json_text)
            return parsed
        except Exception:
            # fall through to line-based parsing
            pass

    # Line-parsing fallback (key: value pairs)
    parsed: Dict[str, Any] = {}
    for line in raw_text.splitlines():
        if ':' in line:
            key, val = line.split(':', 1)
            k = key.strip().lower().replace(' ', '_')
            parsed[k] = val.strip()

    if not parsed:
        parsed = {"llm_raw": raw_text.strip()}

    return parsed


# -----------------------------
#  NORMALIZATION & VALIDATION
# -----------------------------

def normalize_amount(amount_str: str) -> Optional[str]:
    if not amount_str:
        return None
    s = amount_str.replace(',', '').replace('O', '0')
    m = re.search(r'([\$₹€])?\s*([0-9]+(?:\.[0-9]{1,2})?)', s)
    if m:
        symbol = m.group(1) or ''
        number = m.group(2)
        return f"{symbol}{number}"
    return amount_str.strip()


def normalize_date(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    # best-effort normalization: try simple iso-like patterns
    date_str = date_str.strip()
    m = re.search(r'(\d{4}[/-]\d{2}[/-]\d{2})', date_str)
    if m:
        return m.group(1).replace('/', '-')
    m = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})', date_str)
    if m:
        d = m.group(1)
        # attempt to reorder dd/mm/yyyy -> yyyy-mm-dd (naive)
        parts = re.split('[/-]', d)
        return f"{parts[2]}-{parts[1]}-{parts[0]}"
    return date_str


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
#  API ENDPOINTS
# -----------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Enhanced Tesseract OCR API 🚀 (Gemini)"}


@app.post("/extract-receipt")
async def extract_receipt(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Please upload a valid image file (jpg, png, etc.)")

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed_image = preprocess_image(image)
        text = pytesseract.image_to_string(processed_image, lang="eng")

        structured: Dict[str, Any] = {}
        llm_used = False

        # Use Gemini LLM if API key is present
        if GEMINI_API_KEY:
            try:
                llm_used = True
                structured = await call_gemini_llm(text)
            except Exception as e:
                # fallback to regex parser
                structured = parse_receipt_text_regex(text)
                structured.setdefault("_llm_error", str(e))
        else:
            structured = parse_receipt_text_regex(text)

        # normalization
        if 'amount' in structured and isinstance(structured['amount'], str):
            structured['amount_normalized'] = normalize_amount(structured['amount'])
        if 'date' in structured and isinstance(structured['date'], str):
            structured['date_normalized'] = normalize_date(structured['date'])

        # validate via pydantic
        try:
            validated = ReceiptData(**structured)
            structured = validated.dict()
        except ValidationError as ve:
            structured['_validation_error'] = str(ve)

        return {
            "filename": file.filename,
            "llm_used": llm_used,
            "extracted_text": text.strip(),
            "structured_data": structured
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Enhanced OCR API (Gemini) is running ✅"}


# Run locally: uvicorn enhanced_ocr_with_gemini:app --reload
