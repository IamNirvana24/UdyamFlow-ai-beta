"""
Generation layer for the UdyamFlow AI compliance copilot.

Three-tier degradation, which matters because a hackathon demo room has
unreliable wifi and you do not want a blank answer on stage:

  1. LLM available and reachable  -> grounded generated answer
  2. No API key or call fails     -> extractive answer from top chunks
  3. Nothing retrieved            -> honest "not in knowledge base" response

Every answer carries citations back to the knowledge base sections used, so
the copilot can be shown to be grounded rather than hallucinating.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from .retriever import LABEL_TO_SLUG, retriever

# Uses Groq's free tier (OpenAI-compatible endpoint) by default — no credit
# card required, fast enough for a live demo. Get a key at console.groq.com/keys
# and set it as GROQ_API_KEY.
#
# To use a different provider instead, set PROVIDER and swap _call_llm below:
#   - "anthropic": set ANTHROPIC_API_KEY, needs a paid account
#   - "gemini":    set GEMINI_API_KEY, also has a free tier
PROVIDER = os.environ.get("LLM_PROVIDER", "groq").strip().lower()

API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
TIMEOUT_SECONDS = 20

SYSTEM_PROMPT = """You are the compliance copilot inside UdyamFlow AI, a system that helps entrepreneurs in Punjab, India understand which industrial approvals they need.

You are NOT a general chat assistant. You only answer questions about the twelve industrial approvals covered in the reference material. You do not engage in small talk, answer general knowledge questions, or discuss anything outside industrial approvals in Punjab — say so plainly and redirect if asked.

Answer using ONLY the reference material provided in the user's message. Rules:
- If the reference material does not cover the question, say so plainly. Never invent fees, timelines, thresholds, or authority names.
- Numbers matter. If the material says a figure varies or must be confirmed with the department, say exactly that rather than estimating.
- Be concise and practical. Two or three short paragraphs at most. The reader is a business owner, not a lawyer.
- Do not add a disclaimer paragraph at the end; the interface already shows one.
- Write plainly. No bullet lists unless the answer is genuinely a list of documents."""


def _build_context(results) -> str:
    blocks = []
    for i, (chunk, score) in enumerate(results, start=1):
        blocks.append(f"[{i}] {chunk.citation}\n{chunk.text}")
    return "\n\n".join(blocks)


def _call_llm(question: str, context: str) -> str:
    """
    Calls Groq's OpenAI-compatible chat completions endpoint. Groq is free,
    needs no credit card, and is fast enough for a live demo (sub-second on
    llama-3.3-70b-versatile / llama-3.1-8b-instant).
    """
    payload = {
        "model": MODEL,
        "max_tokens": 700,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Reference material:\n\n{context}\n\n"
                    f"---\n\nQuestion: {question}"
                ),
            },
        ],
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        data = json.loads(response.read().decode("utf-8"))

    choices = data.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "").strip()


def _extractive_answer(results) -> str:
    """Fallback: stitch the highest-scoring sections into a readable answer."""
    top = results[0][0]
    body = top.text.split(". ", 2)[-1] if ". " in top.text else top.text
    answer = f"From the {top.title} guidance ({top.section.lower()}):\n\n{body}"

    if len(results) > 1:
        second = results[1][0]
        if second.slug != top.slug:
            answer += f"\n\nThis also touches on {second.title}."
    return answer



# Short greetings and small talk get a friendlier, scope-clarifying reply
# rather than the generic "outside the knowledge base" message — the intent
# here isn't a failed search, it's someone figuring out what this box does.
_GREETING_PATTERN = re.compile(
    r"^\s*(hi|hii+|hello+|hey+|yo|hola|namaste|sup|good\s?(morning|afternoon|evening)|"
    r"how are you|what'?s up|who are you|what can you do|what do you do)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

_SCOPE_STATEMENT = (
    "This copilot only answers questions about the twelve industrial approvals "
    "UdyamFlow predicts — things like what documents an approval needs, why it "
    "applies, or which one to start with. It isn't a general chat assistant. "
    "Try asking something like \"what documents do I need for Fire NOC?\" or "
    "\"which approval should I apply for first?\""
)


def answer_question(question: str, approval: str = None, top_k: int = 4) -> dict:
    """
    Main entry point used by the Flask route.

    approval: optional approval label (e.g. "Fire NOC") when the question is
              asked from a specific card in the approval journey UI.
    """
    if _GREETING_PATTERN.match(question.strip()):
        return {
            "answer": _SCOPE_STATEMENT,
            "citations": [],
            "mode": "scope_notice",
            "grounded": False,
        }

    slug = LABEL_TO_SLUG.get(approval) if approval else None
    results = retriever.search(question, top_k=top_k, approval_filter=slug)

    if not results:
        return {
            "answer": (
                "That is outside what the compliance knowledge base currently covers. "
                "It holds guidance on the twelve approvals UdyamFlow predicts — trade "
                "licence, GST, factory licence, fire and pollution NOCs, environmental "
                "clearance, FSSAI, boiler, explosives, electricity, labour registrations, "
                "and building plan approval. Try rephrasing around one of those."
            ),
            "citations": [],
            "mode": "no_match",
            "grounded": False,
        }

    citations = [
        {
            "title": chunk.title,
            "section": chunk.section,
            "slug": chunk.slug,
            "score": round(score, 3),
        }
        for chunk, score in results
    ]

    context = _build_context(results)

    if API_KEY:
        try:
            generated = _call_llm(question, context)
            if generated:
                return {
                    "answer": generated,
                    "citations": citations,
                    "mode": "generated",
                    "grounded": True,
                }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError):
            # Fall through to extractive rather than failing the request.
            pass

    return {
        "answer": _extractive_answer(results),
        "citations": citations,
        "mode": "extractive",
        "grounded": True,
    }


def suggested_questions(approvals=None):
    """Starter prompts for the UI. Tailored when a prediction has been made."""
    generic = [
        "Which approval should I apply for first?",
        "What is the difference between Consent to Establish and Consent to Operate?",
        "Do I need environmental clearance for a small unit?",
        "What documents do I need for a trade licence?",
    ]
    if not approvals:
        return generic

    tailored = []
    for label in approvals[:3]:
        if label in LABEL_TO_SLUG:
            tailored.append(f"What documents are required for {label}?")
    tailored.append("Which of my approvals will take the longest?")
    return tailored or generic
