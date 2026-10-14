
from src.tools.generate_answer import AnswerGenerator
from src.agents.query_understanding import understand_query
from src.agents.sentiment_agent import analyze_sentiment
from src.memory.conversation_memory import ConversationMemory
from src.feedback.feedback_handler import save_feedback


def test_answer_generator_returns_answer():
    generator = AnswerGenerator()
    result = generator.answer("How can I cancel my order?")

    assert result["answer"]
    assert result["sources"]


def test_unknown_question_uses_fallback():
    generator = AnswerGenerator()
    result = generator.answer("What is the weather today?")

    assert result["answer"]
    assert result["sources"] == []
    assert result["confidence"] == "low"


def test_query_understanding_identifies_cancellation():
    result = understand_query("How can I cancel my order?")

    assert result["intent"] == "cancel_order"


def test_sentiment_agent_detects_negative_message():
    result = analyze_sentiment("My payment failed and I am frustrated.")

    assert result["sentiment"] == "negative"


def test_conversation_memory_stores_messages():
    memory = ConversationMemory()
    memory.add_message("test-session", "user", "Where is my order?")

    history = memory.get_history("test-session")

    assert len(history) == 1
    assert history[0]["content"] == "Where is my order?"


def test_feedback_rejects_invalid_rating():
    try:
        save_feedback(
            question="Where is my order?",
            answer="Please check your tracking details.",
            rating=6,
            session_id="pytest-session",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("An invalid rating should raise ValueError.")
