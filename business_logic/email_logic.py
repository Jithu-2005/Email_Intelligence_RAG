import os
import io
import re
import json
import email
import hashlib
import imaplib
import chromadb
from datetime import datetime
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime, parseaddr, getaddresses
from config import CHROMA_DB_PATH

from dotenv import load_dotenv
from pypdf import PdfReader

from dataaccess.data_models import get_collection

load_dotenv()

BATCH_SIZE = 500

# Ingestion

def _decode(value) -> str:
    return str(make_header(decode_header(value or "")))


def _short_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode()).hexdigest()[:10]


def _get_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            disposition = str(part.get("Content-Disposition") or "")
            if part.get_content_type() == "text/plain" and "attachment" not in disposition:
                payload = part.get_payload(decode=True) or b""
                return payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        return ""
    payload = msg.get_payload(decode=True) or b""
    return payload.decode(msg.get_content_charset() or "utf-8", errors="replace")


def _get_attachments(msg) -> list[dict]:
    attachments = []
    for part in msg.walk():
        filename = part.get_filename()
        if not filename:
            continue
        filename = _decode(filename)
        data = part.get_payload(decode=True) or b""
        text = ""
        try:
            if filename.lower().endswith(".pdf"):
                text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
            elif filename.lower().endswith(".txt"):
                text = data.decode("utf-8", errors="replace")
        except Exception:
            text = ""          
        attachments.append({"filename": filename, "text": text[:5000]})
    return attachments


def _thread_id(msg) -> str:
    refs = (msg.get("References") or "").split()
    root = refs[0] if refs else (msg.get("In-Reply-To") or msg.get("Message-ID") or "")
    return "thr-" + _short_hash(root)


def load_sample(path: str = "data/sample_emails.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_from_imap(host: str = "imap.gmail.com", folder: str = "INBOX", limit: int = 100) -> list[dict]:
    user = os.getenv("IMAP_USER")
    password = os.getenv("IMAP_APP_PASSWORD")
    if not user or not password:
        raise ValueError("Set IMAP_USER and IMAP_APP_PASSWORD in your .env to use the IMAP source.")

    mail = imaplib.IMAP4_SSL(host)
    mail.login(user, password)
    mail.select(folder, readonly=True)                     
    _, data = mail.search(None, "ALL")
    message_numbers = data[0].split()[-limit:]            

    emails = []
    for num in message_numbers:
        _, msg_data = mail.fetch(num, "(BODY.PEEK[])")    
        msg = email.message_from_bytes(msg_data[0][1])
        name, address = parseaddr(msg.get("From", ""))
        emails.append({
            "email_id": "msg-" + _short_hash(msg.get("Message-ID") or num.decode()),
            "thread_id": _thread_id(msg),
            "subject": _decode(msg.get("Subject")),
            "sender": address.lower(),
            "sender_name": _decode(name),
            "recipients": [a.lower() for _, a in getaddresses(msg.get_all("To", []))],
            "date": parsedate_to_datetime(msg["Date"]).isoformat(),
            "labels": [folder],
            "body": _get_body(msg),
            "attachments": _get_attachments(msg),
        })
    mail.logout()
    return emails


def load_emails(source: str = "sample", limit: int = 100) -> list[dict]:
    if source == "sample":
        return load_sample()
    if source == "imap":
        return load_from_imap(limit=limit)
    raise ValueError("source must be 'sample' or 'imap'")


# CHUNKING

QUOTE_HEADER = re.compile(r"^\s*On .{5,150}wrote:\s*$", re.MULTILINE)


def clean_body(body: str) -> str:
    match = QUOTE_HEADER.search(body)
    if match:
        body = body[:match.start()]
    lines = [ln for ln in body.splitlines() if not ln.strip().startswith(">")]
    text = "\n".join(lines)
    text = re.split(r"\n-- \n", text)[0]
    return text.strip()


def split_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def email_to_chunks(e: dict) -> list[dict]:
    header = (f"Subject: {e['subject']}\n"
              f"From: {e.get('sender_name') or e['sender']} <{e['sender']}>\n"
              f"Date: {e['date']}\n\n")

    base_meta = {
        "email_id": e["email_id"], "thread_id": e["thread_id"], "subject": e["subject"],
        "sender": e["sender"].lower(), "sender_name": e.get("sender_name", ""),
        "recipients": ", ".join(e["recipients"]), "date": e["date"],
        "timestamp": int(datetime.fromisoformat(e["date"]).timestamp()),
        "labels": ",".join(e["labels"]), "has_attachment": bool(e.get("attachments")),
    }

    pieces = [("body", p) for p in split_text(clean_body(e["body"]))]
    for att in e.get("attachments", []):
        for p in split_text(att.get("text", "")):
            pieces.append((f"attachment:{att['filename']}", p))
    if not pieces:
        pieces = [("body", "")]        

    return [
        {
            "id": f"{e['email_id']}::{i}",
            "document": header + piece,
            "metadata": {**base_meta, "source": source, "chunk_index": i},
        }
        for i, (source, piece) in enumerate(pieces)
    ]

# INDEXING 

def index_emails(emails: list[dict]) -> int:
    collection = get_collection()
    ids, docs, metas = [], [], []
    for e in emails:
        for c in email_to_chunks(e):
            ids.append(c["id"])
            docs.append(c["document"])
            metas.append(c["metadata"])

    for i in range(0, len(ids), BATCH_SIZE):
        collection.upsert(
            ids=ids[i:i + BATCH_SIZE],
            documents=docs[i:i + BATCH_SIZE],
            metadatas=metas[i:i + BATCH_SIZE],
        )
    return len(ids)

# RETRIEVAL 

def _day_start(d: str) -> int:
    return int(datetime.strptime(d, "%Y-%m-%d").timestamp())


def _day_end(d: str) -> int:
    return _day_start(d) + 86399


def search_emails(query: str, top_k: int = 5, sender: str = "",
                    after: str = "", before: str = "", label: str = "") -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    clauses = []
    if after:
        clauses.append({"timestamp": {"$gte": _day_start(after)}})
    if before:
        clauses.append({"timestamp": {"$lte": _day_end(before)}})
    where = None if not clauses else (clauses[0] if len(clauses) == 1 else {"$and": clauses})

    over_fetch = top_k * 4 if (sender or label) else top_k
    res = collection.query(
        query_texts=[query],
        n_results=min(over_fetch, collection.count()),
        where=where,
    )

    results = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if sender and sender.lower() not in f"{meta['sender']} {meta.get('sender_name', '')}".lower():
            continue
        if label and label.lower() not in [l.strip().lower() for l in meta.get("labels", "").split(",")]:
            continue
        results.append({"text": text, "metadata": meta, "distance": dist})
    return results[:top_k]