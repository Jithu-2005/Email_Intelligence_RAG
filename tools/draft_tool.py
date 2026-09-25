from config.config import llm, MY_EMAIL
from dataaccess.data_models import get_collection, get_thread_text
from business_logic.email_logic import DRAFT_PROMPT
from langchain_core.tools import tool


def _style_samples(n: int = 3) -> str:
    """A few emails the user has previously written, so drafts sound like them."""
    if not MY_EMAIL:
        return "No past emails available. Use a neutral, polite tone."
    res = get_collection().get(where={"sender": MY_EMAIL}, limit=n)
    return "\n---\n".join(res["documents"]) or "No past emails available. Use a neutral, polite tone."


@tool
def draft_tool(thread_id: str, intent: str, tone: str = "professional") -> str:
    """
    Use this tool to DRAFT A REPLY to an email thread. It only writes the draft text —
    it can never send an email.
    You MUST have the thread_id from a previous search_tool result — never guess one.

    thread_id: the thread to reply to, e.g. "thr-001" — required
    intent: what the reply should say or achieve, e.g. "confirm the 23 Sept meeting" — required
    tone: "professional" | "friendly" | "formal" | "brief" — optional, default "professional"
    """
    thread = get_thread_text(thread_id)
    if not thread:
        return f"No thread found with id '{thread_id}'. Use search_tool first to find the right thread_id."

    prompt = DRAFT_PROMPT.format(
        intent=intent, tone=tone, style_samples=_style_samples(), thread=thread
    )
    return llm.invoke(prompt).content
