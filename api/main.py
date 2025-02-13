import os
import json
import base64
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File
from mangum import Mangum
import google.generativeai as genai

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("business-card-api")

# Initialize the FastAPI app
app = FastAPI(title="Business Card Extraction API")

# Retrieve the API key from the environment (fallback provided for testing)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "youAPI")
if not GOOGLE_API_KEY:
    logger.error("Missing GOOGLE_API_KEY environment variable.")
    raise Exception("Missing GOOGLE_API_KEY environment variable.")
genai.configure(api_key=GOOGLE_API_KEY)

# Define the prompt for the Gemini model
EXTRACTION_PROMPT = """
Analyze the business card and extract the following information in JSON format:
{
  "name": "Full Name",
  "mobile_number": "Mobile Number",
  "company_name": "Company Name",
  "email": "Email Address",
  "job_title": "Job Title",
  "address": "Address",
  "website": "Website URL"
}

Ensure that the response is strictly in JSON format and includes all the fields mentioned above. If any information is missing or unclear, set the corresponding field to null.
"""

def remove_markdown_formatting(text: str) -> str:
    """
    Remove markdown formatting (e.g. triple-backticks) from the response.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped

@app.get("/status")
async def status():
    """
    Health-check endpoint to verify that the API is running.
    """
    return {"status": "ok"}

@app.post("/upload")
async def extract_business_card(file: UploadFile = File(...)):
    """
    Accept an uploaded image file, extract business card details using the Gemini model,
    and return the extracted information as JSON.
    """
    # Validate that the uploaded file is a JPEG or PNG image.
    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.warning("Unsupported file type: %s", file.content_type)
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG and PNG images are supported."
        )

    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.exception("Error reading uploaded file")
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    encoded_image = base64.b64encode(file_bytes).decode("utf-8")

    # Initialize the Gemini model.
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    except Exception as e:
        logger.exception("Error initializing Gemini model")
        raise HTTPException(status_code=500, detail=f"Error initializing model: {e}")

    # Generate the content based on the image and prompt.
    try:
        response = model.generate_content([
            {"mime_type": file.content_type, "data": encoded_image},
            EXTRACTION_PROMPT,
        ])
    except Exception as e:
        logger.exception("Error during content generation")
        raise HTTPException(status_code=500, detail=f"Error generating content: {e}")

    # Remove markdown formatting if present.
    raw_text = remove_markdown_formatting(response.text)
    logger.info("Raw response (first 100 chars): %s", raw_text[:100])

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as jde:
        logger.exception("JSON decoding error")
        raise HTTPException(
            status_code=500,
            detail=f"JSON decoding error: {jde}. Full response was: {response.text}"
        )

    return data

# Create a Mangum handler to adapt the FastAPI app for Vercel's serverless environment.
handler = Mangum(app)
