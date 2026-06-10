import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.agent import route_question
from app.rag import ingest_documents
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Healthcare AI Assistant",
    description="A RAG-based AI assistant for answering healthcare-related questions from documents.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_ui():
    return FileResponse("chat.html")


class QuestionRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    data_dir: str = None


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Healthcare AI Assistant",
        "version": "1.0.0"
    }


@app.post("/ingest")
def ingest(request: IngestRequest = None):
    try:
        data_dir = (request.data_dir if request else None) or settings.data_dir
        logger.info(f"Starting document ingestion from: {data_dir}")

        result = ingest_documents(data_dir)

        logger.info(f"Ingestion complete: {result['total_chunks']} chunks from {len(result['files_processed'])} files")

        return {
            "status": "success",
            "message": f"Ingested {result['total_chunks']} chunks from {len(result['files_processed'])} files",
            "details": result
        }

    except FileNotFoundError as e:
        logger.error(f"Data directory not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ask")
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        logger.info(f"Question received: {request.question}")
        result = route_question(request.question)
        logger.info(f"Answer generated | tool={result.get('tool_used')} | confidence={result.get('confidence')}")
        return result

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")
