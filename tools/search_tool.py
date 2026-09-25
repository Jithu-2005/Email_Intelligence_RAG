from langchain_core.tools import tool
from business_logic.email_logic import search_emails

MAX_CHARS = 6000


@tool
def search_tool(query: str, top_k: int = 5, sender: str = "", after: str = "",
                before: str = "", label: str = "") -> str:
    """
    Use this tool to FIND information inside the user's emails: what was said, decided,
    requested, or scheduled. Call this FIRST for any email question, and before summarizing,
    extracting from, or drafting a reply to a thread, because it returns the thread_id
    those other tools need.

    query: what to look for, in plain words (e.g. "action items from client sync")
    top_k: how many email chunks to return — optional, default 5
    sender: only emails from this person (name or email address) — optional
    after: only emails on or after this date, format YYYY-MM-DD — optional
    before: only emails on or before this date, format YYYY-MM-DD — optional
    label: only emails with this label, e.g. "INBOX" or "finance" — optional
    """
    try:
        results = search_emails(query, top_k, sender, after, before, label)
    except Exception as e:
        return f"Email search failed: {e}"
    if not results:
        return "No relevant emails found."

    blocks = []
    for r in results:
        m = r["metadata"]
        blocks.append(f"[{m['email_id']}] (thread: {m['thread_id']})\n{r['text']}")
    return "\n\n---\n\n".join(blocks)[:MAX_CHARS]