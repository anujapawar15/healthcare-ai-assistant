# Healthcare AI Assistant

A RAG-based AI assistant that answers healthcare-related questions from internal documents. Built with FastAPI, ChromaDB, sentence-transformers, and Groq (LLaMA 3.3).

---

## Architecture

```
User Question
     │
     ▼
┌──────────────────────────────┐
│        FastAPI  /ask         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Agent (Router)          │
│  Appointment keyword? ──Yes──┼──► Mock Appointment Tool
│           │ No               │
└───────────┼──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│       RAG Pipeline           │
│  1. Embed the question       │
│  2. Query ChromaDB           │
│  3. Retrieve top-K chunks    │
│  4. Build context            │
│  5. Call LLM                 │
└──────────────────────────────┘
            │
            ▼
    Grounded Answer + Sources
```

---

## Tech Stack

| Component       | Tool                      | Why                                    |
|----------------|---------------------------|----------------------------------------|
| API Framework   | FastAPI                   | Fast, modern, auto-generates docs      |
| Vector Database | ChromaDB                  | No external server, runs locally       |
| Embeddings      | sentence-transformers     | Free, runs on CPU, high quality        |
| Embedding Model | all-MiniLM-L6-v2          | Small (80MB), fast, good accuracy      |
| LLM             | Groq — LLaMA 3.3 70B      | Free tier, very fast inference         |
| Configuration   | pydantic-settings + .env  | Clean, no hardcoded secrets            |
| Chat UI         | Vanilla HTML/CSS/JS       | No framework needed, single file       |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/healthcare-ai-assistant.git
cd healthcare-ai-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
```
Edit `.env` and add your Groq API key:
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at https://console.groq.com → API Keys

### 5. Run the server
```bash
python run.py
```
Server starts at http://localhost:8000

### 6. Ingest documents (one-time setup)
```bash
curl -X POST http://localhost:8000/ingest
```

### 7. Open the Chat UI
Visit **http://localhost:8000** in your browser.

---

## Running with Docker

```bash
cp .env.example .env
# Add your API key to .env

docker-compose up --build
curl -X POST http://localhost:8000/ingest
```

---

## API Endpoints

### GET /health
```bash
curl http://localhost:8000/health
```
```json
{"status": "healthy", "service": "Healthcare AI Assistant", "version": "1.0.0"}
```

### POST /ingest
Reads all `.txt` files from the `data/` folder and stores embeddings in ChromaDB.
```bash
curl -X POST http://localhost:8000/ingest
```
```json
{
  "status": "success",
  "message": "Ingested 26 chunks from 6 files",
  "details": {
    "files_processed": ["discharge_instructions.txt", "telehealth_guidelines.txt", "..."],
    "total_chunks": 26
  }
}
```

### POST /ask
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can a patient request a medication refill through telehealth?"}'
```
```json
{
  "answer": "Yes, medication refill requests may be reviewed during telehealth visits for certain non-controlled medications...",
  "sources": [
    {
      "document": "telehealth_guidelines.txt",
      "chunk": "Medication refill requests may be reviewed during telehealth visits..."
    }
  ],
  "confidence": "high",
  "tool_used": "rag_knowledge_base"
}
```

---

## Prompt Engineering Strategy

The system prompt is designed around strict grounding — the model must only answer from provided context and never guess.

```
You are a healthcare information assistant for a medical facility. Your role is to answer
questions using only the information provided in the context below.

Rules:
- Answer only from the provided context. Do not use any outside knowledge.
- If the answer is not found, respond: "I could not find this information in the provided documents."
- Never diagnose medical conditions or recommend specific treatments.
- Keep responses professional, clear, and concise.
```

---

## Agent/Tool Workflow

The agent is a **keyword-based router** — simple, deterministic, and explainable.

```
incoming question
      │
      ▼
Does question contain: "book", "schedule", "appointment", "slot", "available"?
      │                                │
     Yes                               No
      │                                │
      ▼                                ▼
check_available_slots()           RAG Pipeline
(mock appointment tool)           (document Q&A)
```

---

## Sample Questions

| Question | Tool Used | Confidence |
|----------|-----------|------------|
| Can a patient get a medication refill through telehealth? | RAG | high |
| What are the warning signs after discharge? | RAG | high |
| What is HIPAA and what are my rights? | RAG | high |
| Can I book a cardiology appointment for Monday? | appointment_scheduler | high |
| What is the meaning of life? | RAG | none (not in docs) |

---

## Limitations

- Only knows what is in the `/data` folder
- Appointment tool is a mock — no real calendar integration
- ChromaDB stores data locally — not suitable for multi-instance deployments

## Future Improvements

- LLM-based intent classification for smarter routing
- PDF and DOCX document ingestion
- Multi-turn conversation history
- Healthcare-specific embedding model (e.g., Bio_ClinicalBERT)
- Re-ranking step after retrieval
- Real appointment scheduling API integration

---

## Healthcare Data Privacy Notes

- All documents in `/data` are **synthetic** — no real patient data, no PHI
- System prompt prevents the model from providing medical diagnoses
- In production: use HTTPS, avoid logging PHI, deploy in HIPAA-compliant environment
