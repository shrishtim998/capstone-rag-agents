
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from src.agents.query_understanding import understand_query
from src.agents.sentiment_agent import analyze_sentiment
from src.agents.validation_agent import validate_response
from src.agents.escalation_agent import decide_escalation
from src.tools.generate_answer import AnswerGenerator


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_IMAGE_PATH = PROJECT_ROOT / "artifacts" / "support_graph.png"


# --------------------------------------------------
# 2. Define graph state
# --------------------------------------------------

class SupportState(TypedDict, total=False):
    question: str
    query_info: dict[str, Any]
    sentiment_info: dict[str, Any]
    answer_result: dict[str, Any]
    validation: dict[str, Any]
    escalation: dict[str, Any]
    final_response: dict[str, Any]


# --------------------------------------------------
# 3. Initialize answer generator
# --------------------------------------------------

answer_generator = AnswerGenerator()


# --------------------------------------------------
# 4. Define graph nodes
# --------------------------------------------------

def query_understanding_node(state: SupportState) -> dict:
    result = understand_query(state["question"])
    return {"query_info": result}


def sentiment_node(state: SupportState) -> dict:
    result = analyze_sentiment(state["question"])
    return {"sentiment_info": result}


def support_node(state: SupportState) -> dict:
    result = answer_generator.answer(state["question"])
    return {"answer_result": result}


def validation_node(state: SupportState) -> dict:
    result = validate_response(state["answer_result"])
    return {"validation": result}


def escalation_node(state: SupportState) -> dict:
    result = decide_escalation(
        intent=state["query_info"]["intent"],
        sentiment=state["sentiment_info"]["sentiment"],
        validation=state["validation"],
    )
    return {"escalation": result}


def human_support_node(state: SupportState) -> dict:
    question = state["question"]
    query_info = state.get("query_info", {})
    sentiment_info = state.get("sentiment_info", {})

    response = {
        "question": question,
        "answer": (
            "I’m unable to resolve this request automatically. "
            "Please contact customer support for further assistance."
        ),
        "sources": [],
        "confidence": "low",
        "intent": query_info.get("intent", "unknown"),
        "category": query_info.get("category", "UNKNOWN"),
        "sentiment": sentiment_info.get("sentiment", "unknown"),
        "priority": sentiment_info.get("priority", "normal"),
        "recommended_action": "human_support",
        "needs_escalation": True,
    }

    return {"final_response": response}


def format_response_node(state: SupportState) -> dict:
    result = state["answer_result"]
    query_info = state["query_info"]
    sentiment_info = state["sentiment_info"]
    validation = state["validation"]
    escalation = state["escalation"]

    response = {
        **result,
        "intent": query_info.get("intent", "unknown"),
        "category": query_info.get("category", "UNKNOWN"),
        "sentiment": sentiment_info.get("sentiment", "unknown"),
        "priority": sentiment_info.get("priority", "normal"),
        "validation": validation,
        "escalation": escalation,
        "recommended_action": escalation.get(
            "recommended_action",
            "automated_support",
        ),
        "needs_escalation": escalation.get(
            "needs_escalation",
            False,
        ),
    }

    return {"final_response": response}


# --------------------------------------------------
# 5. Conditional routing functions
# --------------------------------------------------

def route_after_understanding(state: SupportState) -> str:
    intent = state["query_info"].get("intent", "unknown")

    if intent == "unknown":
        return "human_support"

    return "analyze_sentiment"


def route_after_validation(state: SupportState) -> str:
    validation = state["validation"]

    if not validation.get("is_valid", False):
        return "human_support"

    return "decide_escalation"


def route_after_escalation(state: SupportState) -> str:
    escalation = state["escalation"]

    if escalation.get("needs_escalation", False):
        return "human_support"

    return "format_response"


# --------------------------------------------------
# 6. Build the LangGraph workflow
# --------------------------------------------------

def build_support_graph():
    graph = StateGraph(SupportState)

    graph.add_node("understand_query", query_understanding_node)
    graph.add_node("analyze_sentiment", sentiment_node)
    graph.add_node("generate_support", support_node)
    graph.add_node("validate_response", validation_node)
    graph.add_node("decide_escalation", escalation_node)
    graph.add_node("human_support", human_support_node)
    graph.add_node("format_response", format_response_node)

    graph.add_edge(START, "understand_query")

    graph.add_conditional_edges(
        "understand_query",
        route_after_understanding,
        {
            "analyze_sentiment": "analyze_sentiment",
            "human_support": "human_support",
        },
    )

    graph.add_edge("analyze_sentiment", "generate_support")
    graph.add_edge("generate_support", "validate_response")

    graph.add_conditional_edges(
        "validate_response",
        route_after_validation,
        {
            "decide_escalation": "decide_escalation",
            "human_support": "human_support",
        },
    )

    graph.add_conditional_edges(
        "decide_escalation",
        route_after_escalation,
        {
            "human_support": "human_support",
            "format_response": "format_response",
        },
    )

    graph.add_edge("human_support", END)
    graph.add_edge("format_response", END)

    return graph.compile()


# --------------------------------------------------
# 7. Compile the graph
# --------------------------------------------------

support_graph = build_support_graph()


# --------------------------------------------------
# 8. Run tests and save the graph image
# --------------------------------------------------

if __name__ == "__main__":
    test_questions = [
        "How can I cancel my order?",
        "My order has not arrived yet.",
        "My payment failed and I am very frustrated.",
        "What is the weather today?",
    ]

    for question in test_questions:
        print("\n" + "=" * 70)
        print("QUESTION:", question)

        try:
            result = support_graph.invoke({"question": question})
            print("\nFINAL RESPONSE:")
            print(result["final_response"])

        except Exception as exc:
            print(f"Error processing question: {exc}")

    try:
        GRAPH_IMAGE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        graph_png = support_graph.get_graph().draw_mermaid_png()
        GRAPH_IMAGE_PATH.write_bytes(graph_png)

        print(f"\nVisual graph saved to: {GRAPH_IMAGE_PATH}")

    except Exception as exc:
        print(f"\nCould not generate graph image: {exc}")
