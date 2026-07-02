from flask import Flask, request, jsonify, send_from_directory
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

app = Flask(__name__, static_folder=None)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "src"

ENDPOINT = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
KEY = os.environ.get("LANGUAGE_KEY", "")

API_VERSION = "2023-04-01"
SUMMARIZATION_API_VERSION = "2023-04-01"
TIMEOUT_SECONDS = 20
SUMMARIZATION_POLL_INTERVAL_SECONDS = 1
SUMMARIZATION_MAX_WAIT_SECONDS = 60
MAX_CHARS = 5000


@app.get("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(FRONTEND_DIR), filename)


@app.post("/api/analyze")
def analyze():
    endpoint = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
    key = os.environ.get("LANGUAGE_KEY", "")
    if not endpoint or not key:
        return jsonify({"error": "Server is missing LANGUAGE_ENDPOINT / LANGUAGE_KEY environment variables."}), 500

    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": 'Request body must include non-empty "text".'}), 400
    if len(text) > MAX_CHARS:
        return jsonify({"error": f"Text must be {MAX_CHARS} characters or fewer."}), 400

    try:
        sentiment_doc = _call_language("SentimentAnalysis", text)["results"]["documents"][0]
        keyphrase_doc = _call_language("KeyPhraseExtraction", text)["results"]["documents"][0]
        entity_doc = _call_language("EntityRecognition", text)["results"]["documents"][0]
    except Exception:
        logging.exception("Azure AI Language call failed")
        return jsonify({"error": "Azure AI Language request failed. Check key/endpoint/quota."}), 502

    language_info = {}
    try:
        language_doc = _call_language("LanguageDetection", text, language=None)["results"]["documents"][0]
        language_info = language_doc.get("detectedLanguage", {})
    except Exception:
        logging.exception("Language detection failed")

    pii_entities = []
    redacted_text = ""
    try:
        pii_doc = _call_language("PiiEntityRecognition", text, language=None)["results"]["documents"][0]
        pii_entities = [
            {"text": e["text"], "category": e["category"]}
            for e in pii_doc.get("entities", [])
        ]
        redacted_text = pii_doc.get("redactedText", "")
    except Exception:
        logging.exception("PII detection failed")

    summary_text = ""
    try:
        summary_text = _call_abstractive_summarization(text)
    except Exception:
        logging.exception("Abstractive summarization failed")

    result = {
        "sentiment": sentiment_doc["sentiment"],
        "confidenceScores": sentiment_doc["confidenceScores"],
        "keyPhrases": keyphrase_doc.get("keyPhrases", []),
        "entities": [
            {"text": e["text"], "category": e["category"]}
            for e in entity_doc.get("entities", [])
        ],
        "language": language_info,
        "piiEntities": pii_entities,
        "redactedText": redacted_text,
        "summary": summary_text,
    }
    return jsonify(result)


def _call_language(kind: str, text: str, *, language: str = "en") -> dict:
    url = f"{ENDPOINT}/language/:analyze-text?api-version={API_VERSION}"
    document = {"id": "1", "text": text}
    if language:
        document["language"] = language
    payload = {
        "kind": kind,
        "parameters": {"modelVersion": "latest"},
        "analysisInput": {"documents": [document]},
    }
    return _request_json("POST", url, payload)


def _call_abstractive_summarization(text: str) -> str:
    submit_url = f"{ENDPOINT}/language/analyze-text/jobs?api-version={SUMMARIZATION_API_VERSION}"
    payload = {
        "displayName": "Text Abstractive Summarization",
        "analysisInput": {
            "documents": [{"id": "1", "language": "en", "text": text}],
        },
        "tasks": [
            {
                "kind": "AbstractiveSummarization",
                "taskName": "summarize",
            }
        ],
    }
    _, headers = _request_json("POST", submit_url, payload, return_headers=True)
    status_url = headers.get("operation-location") or headers.get("Operation-Location")
    if not status_url:
        raise RuntimeError("Summarization job did not return operation-location header.")

    deadline = time.monotonic() + SUMMARIZATION_MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        job = _request_json("GET", status_url)
        status = job.get("status", "")
        if status == "succeeded":
            return _extract_summary_text(job)
        if status in ("failed", "cancelled"):
            errors = job.get("errors") or []
            raise RuntimeError(f"Summarization job {status}: {errors}")
        time.sleep(SUMMARIZATION_POLL_INTERVAL_SECONDS)

    raise RuntimeError("Summarization job timed out.")


def _extract_summary_text(job: dict) -> str:
    items = (job.get("tasks") or {}).get("items") or []
    for item in items:
        documents = (item.get("results") or {}).get("documents") or []
        for document in documents:
            summaries = document.get("summaries") or []
            if summaries:
                return summaries[0].get("text", "")
    return ""


def _request_json(method: str, url: str, payload: dict | None = None, *, return_headers: bool = False):
    data = None
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": KEY,
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            if return_headers:
                return body, dict(resp.headers)
            return body
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {detail}") from exc


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
    endpoint = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
    key = os.environ.get("LANGUAGE_KEY", "")
    if not endpoint or not key:
        return jsonify({"error": "Server is missing LANGUAGE_ENDPOINT / LANGUAGE_KEY environment variables."}), 500

    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": 'Request body must include non-empty "text".'}), 400
    if len(text) > MAX_CHARS:
        return jsonify({"error": f"Text must be {MAX_CHARS} characters or fewer."}), 400

    try:
        sentiment_doc = _call_language("SentimentAnalysis", text)["results"]["documents"][0]
        keyphrase_doc = _call_language("KeyPhraseExtraction", text)["results"]["documents"][0]
        entity_doc = _call_language("EntityRecognition", text)["results"]["documents"][0]
    except Exception:
        logging.exception("Azure AI Language call failed")
        return jsonify({"error": "Azure AI Language request failed. Check key/endpoint/quota."}), 502

    language_info = {}
    try:
        language_doc = _call_language("LanguageDetection", text, language=None)["results"]["documents"][0]
        language_info = language_doc.get("detectedLanguage", {})
    except Exception:
        logging.exception("Language detection failed")

    pii_entities = []
    redacted_text = ""
    try:
        pii_doc = _call_language("PiiEntityRecognition", text, language=None)["results"]["documents"][0]
        pii_entities = [
            {"text": e["text"], "category": e["category"]}
            for e in pii_doc.get("entities", [])
        ]
        redacted_text = pii_doc.get("redactedText", "")
    except Exception:
        logging.exception("PII detection failed")

    summary_text = ""
    try:
        summary_text = _call_abstractive_summarization(text)
    except Exception:
        logging.exception("Abstractive summarization failed")

    result = {
        "sentiment": sentiment_doc["sentiment"],
        "confidenceScores": sentiment_doc["confidenceScores"],
        "keyPhrases": keyphrase_doc.get("keyPhrases", []),
        "entities": [
            {"text": e["text"], "category": e["category"]}
            for e in entity_doc.get("entities", [])
        ],
        "language": language_info,
        "piiEntities": pii_entities,
        "redactedText": redacted_text,
        "summary": summary_text,
    }
    return jsonify(result)


def _call_language(kind: str, text: str, *, language: str = "en") -> dict:
    url = f"{ENDPOINT}/language/:analyze-text?api-version={API_VERSION}"
    document = {"id": "1", "text": text}
    if language:
        document["language"] = language
    payload = {
        "kind": kind,
        "parameters": {"modelVersion": "latest"},
        "analysisInput": {"documents": [document]},
    }
    return _request_json("POST", url, payload)


def _call_abstractive_summarization(text: str) -> str:
    submit_url = f"{ENDPOINT}/language/analyze-text/jobs?api-version={SUMMARIZATION_API_VERSION}"
    payload = {
        "displayName": "Text Abstractive Summarization",
        "analysisInput": {
            "documents": [{"id": "1", "language": "en", "text": text}],
        },
        "tasks": [
            {
                "kind": "AbstractiveSummarization",
                "taskName": "summarize",
            }
        ],
    }
    _, headers = _request_json("POST", submit_url, payload, return_headers=True)
    status_url = headers.get("operation-location") or headers.get("Operation-Location")
    if not status_url:
        raise RuntimeError("Summarization job did not return operation-location header.")

    deadline = time.monotonic() + SUMMARIZATION_MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        job = _request_json("GET", status_url)
        status = job.get("status", "")
        if status == "succeeded":
            return _extract_summary_text(job)
        if status in ("failed", "cancelled"):
            errors = job.get("errors") or []
            raise RuntimeError(f"Summarization job {status}: {errors}")
        time.sleep(SUMMARIZATION_POLL_INTERVAL_SECONDS)

    raise RuntimeError("Summarization job timed out.")


def _extract_summary_text(job: dict) -> str:
    items = (job.get("tasks") or {}).get("items") or []
    for item in items:
        documents = (item.get("results") or {}).get("documents") or []
        for document in documents:
            summaries = document.get("summaries") or []
            if summaries:
                return summaries[0].get("text", "")
    return ""


def _request_json(method: str, url: str, payload: dict | None = None, *, return_headers: bool = False):
    data = None
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": KEY,
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            if return_headers:
                return body, dict(resp.headers)
            return body
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {detail}") from exc


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
