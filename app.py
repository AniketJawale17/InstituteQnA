# APIs for Institute QnA application

from fastapi import FastAPI
import os
import uvicorn
from dotenv import load_dotenv
from institute_qna import WebBasedLoader
from institute_qna.data_preprocess.extract_pdf_text import PDFTextExtractor
from institute_qna.logging_config import configure_logging
import pandas as pd

load_dotenv()

# Configure application logging (fall back to logs/app.log)
log_file = os.getenv("LOGGER_FILE") or "logs/app.log"
configure_logging(level=os.getenv("LOG_LEVEL", "INFO"), log_file=log_file)

app = FastAPI(
    title="Institutional Question Anwering System",
    description="Question Answering system on COEP Admission data",
    version = '0.1.0',
    # contact="aniketjawale17@gmail.com"
    )

import logging
logger = logging.getLogger(__name__)



@app.get("/")
async def system_status() -> dict:
    """Provides the system status as up"""
    logger.info("App started successfully")
    return {
        "Status": "Up and Running"
    }


@app.post("/Extract")

async def extract() -> dict:
    
    try:
        url = "https://www.coeptech.ac.in/admissions/undergraduate/"
        WebBasedLoader.load_html_markdown_from_url(url)
        logger.info("Generated admission data json in extracted text data")
        return {"Status": "Success"}
    except Exception as e:
        # Log full stack and return a 500-like message
        logger.exception("Unexpected error occurred while extracting data",e)
        # Re-raise so FastAPI returns a 500 response (or you can return a custom response)
        return {"Error" : str(e)}
    

@app.post("/Process")
async def process(pdf_path_folder : str) -> dict:
    try:
        # Call the PDF text extraction method
        extracted_text = []
        for pdf_file in os.listdir(pdf_path_folder):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_path_folder, pdf_file)
                extracted_text += PDFTextExtractor.extract_text_from_text_pdf(pdf_path)
                logger.info("Extracted text from text-based PDF: %s", pdf_file)
        df = pd.DataFrame([docs.page_content for docs in extracted_text], columns=["Extracted_Text"])
        df.to_csv("extracted_text_data/extracted_pdf_text.csv", index=False)
        logger.info("Saved extracted text to CSV file")
        return {"Status": "Success", "Data": extracted_text}
    except Exception as e:
        logger.exception("Unexpected error occurred while processing PDF", e)
        return {"Error": str(e)}


if __name__ == "__main__":
    uvicorn.run("app:app", reload=False, workers=4, port=8005, host="0.0.0.0")
