import logging
from app.rag import ask_question

logger = logging.getLogger(__name__)

APPOINTMENT_KEYWORDS = [
    "book", "schedule", "appointment", "slot", "available", "availability",
    "booking", "reserve", "when can i come", "next visit", "walk-in"
]

MOCK_SLOTS = {
    "cardiology": ["Monday 10:00 AM", "Monday 2:30 PM", "Wednesday 9:00 AM"],
    "neurology": ["Tuesday 11:00 AM", "Thursday 3:00 PM", "Friday 9:30 AM"],
    "orthopedics": ["Wednesday 10:00 AM", "Friday 9:00 AM", "Thursday 2:00 PM"],
    "pediatrics": ["Monday 11:00 AM", "Tuesday 2:00 PM", "Friday 10:00 AM"],
    "dermatology": ["Wednesday 3:00 PM", "Thursday 10:00 AM", "Monday 4:00 PM"],
    "general": ["Monday 9:00 AM", "Tuesday 10:00 AM", "Wednesday 2:00 PM", "Thursday 11:00 AM"],
}


def is_appointment_question(question: str) -> bool:
    return any(keyword in question.lower() for keyword in APPOINTMENT_KEYWORDS)


def extract_department(question: str) -> str:
    question_lower = question.lower()
    for dept in MOCK_SLOTS:
        if dept in question_lower:
            return dept
    return "general"


def check_available_slots(department: str) -> dict:
    slots = MOCK_SLOTS.get(department.lower(), MOCK_SLOTS["general"])
    return {
        "department": department.capitalize(),
        "available_slots": slots
    }


def route_question(question: str) -> dict:
    logger.info(f"Agent received question: {question}")

    if is_appointment_question(question):
        logger.info("Routing to appointment tool")
        department = extract_department(question)
        slot_info = check_available_slots(department)

        answer = (
            f"I can check mock appointment availability for {slot_info['department']}. "
            f"Available slots this week: {', '.join(slot_info['available_slots'])}. "
            f"To confirm your booking, please call (555) 000-1234 or use the patient portal."
        )

        return {
            "answer": answer,
            "sources": [],
            "confidence": "high",
            "tool_used": "appointment_scheduler",
            "tool_result": slot_info
        }

    logger.info("Routing to RAG knowledge base")
    result = ask_question(question)
    result["tool_used"] = "rag_knowledge_base"
    return result
