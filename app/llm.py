import logging
from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a healthcare information assistant for a medical facility. Your role is to answer questions using only the information provided in the context below.

You must follow these rules strictly:
- Answer only from the provided context. Do not use any outside knowledge or assumptions.
- If the answer is not found in the context, respond with exactly: "I could not find this information in the provided documents."
- Never diagnose medical conditions or recommend specific treatments.
- Never prescribe or advise on medications beyond what is explicitly stated in the documents.
- Keep responses professional, clear, and concise.
- Do not speculate or infer beyond what is explicitly written in the context."""


def generate_answer(question: str, context: str) -> str:
    prompt = f"""{SYSTEM_PROMPT}

Context from healthcare documents:
{context}

Question: {question}

Answer (based only on the context above):"""

    if settings.llm_provider == "openai":
        return _call_openai(prompt)
    if settings.llm_provider == "claude":
        return _call_claude(prompt)
    if settings.llm_provider == "huggingface":
        return _call_huggingface(prompt)
    if settings.llm_provider == "groq":
        return _call_groq(prompt)
    return _call_gemini(prompt)


def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


def _call_claude(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def _call_huggingface(prompt: str) -> str:
    import httpx
    url = f"https://router.huggingface.co/hf-inference/models/{settings.huggingface_model}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.huggingface_api_key}"}
    payload = {
        "model": settings.huggingface_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512
    }
    response = httpx.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _call_groq(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content


def _call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content
