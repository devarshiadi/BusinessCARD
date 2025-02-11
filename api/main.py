from fastapi import FastAPI, HTTPException, UploadFile, File
import google.generativeai as genai
import base64
import json
from mangum import Mangum

# Initialize the FastAPI app
app = FastAPI(title="Business Card Extraction API")

# Replace with your actual Google API Key
GOOGLE_API_KEY = 'AIzaSyDcYyq3w21iwipYn17wCAQo3AYWhUIGDSI'
genai.configure(api_key=GOOGLE_API_KEY)

# Define the prompt instructing the model what information to extract
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

@app.get("/status")
def status():
    """
    Health-check endpoint to verify the API is running.
    """
    return {"status": "ok"}

@app.post("/upload")
async def extract_business_card(file: UploadFile = File(...)):
    """
    Accepts an image file upload, extracts business card information using the Gemini model,
    and returns the extracted data as JSON.
    """
    # Validate file type (accepting JPEG and PNG images)
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG and PNG images are supported."
        )

    # Read the uploaded file content
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    # Encode the image in base64
    encoded_image = base64.b64encode(file_bytes).decode("utf-8")

    # Create an instance of the Gemini model (adjust model name if necessary)
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    # Call the model with the encoded image and extraction prompt
    try:
        response = model.generate_content([
            {
                "mime_type": file.content_type,  # "image/jpeg" or "image/png"
                "data": encoded_image,
            },
            EXTRACTION_PROMPT,
        ])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating content: {e}")

    # Get the raw text response and remove markdown formatting if present
    raw = response.text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        # Remove starting markdown (e.g., ```json)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove ending markdown (```), if present
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # Parse the JSON from the cleaned text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as jde:
        raise HTTPException(
            status_code=500,
            detail=f"JSON decoding error: {jde}. Full response was: {response.text}"
        )

    return data

# Create a Mangum handler for Vercel's serverless environment
handler = Mangum(app)

# When testing locally, you can run:
# uvicorn api.main:app --reload
