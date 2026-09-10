#!/usr/bin/env python3
"""Local web UI wrapping the same pipeline run.py drives from the CLI.

    python webapp.py

Then open http://127.0.0.1:5050 . This is a thin Flask layer over
extraction_agent.py / research_agent.py — no agent logic lives here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI

from run import run_pipeline
from run_logger import RunLogger

load_dotenv()

app = Flask(__name__)
EVAL_DIR = Path(__file__).parent / "eval"
LOG_DIR = Path(__file__).parent / "logs"


@app.get("/")
def index():
    samples = []
    for p in sorted(EVAL_DIR.glob("sample*.txt")):
        samples.append({"name": p.name, "text": p.read_text(encoding="utf-8").strip()})
    return render_template("index.html", samples=samples)


@app.post("/api/run")
def api_run():
    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify({"error": "OPENAI_API_KEY is not set (check your .env file)."}), 500

    data = request.get_json(silent=True) or {}
    disclosure_text = (data.get("disclosure_text") or "").strip()
    if not disclosure_text:
        return jsonify({"error": "Paste or select a disclosure first."}), 400

    client = OpenAI()
    logger = RunLogger(log_dir=str(LOG_DIR))
    try:
        extraction, report = run_pipeline(disclosure_text, logger, client)
        summary = logger.finalize()
    except RuntimeError as e:
        logger.log_error(str(e))
        logger.finalize()
        return jsonify({"error": str(e), "log": logger.path.read_text(encoding="utf-8")}), 500

    return jsonify(
        {
            "extraction": extraction.model_dump(),
            "report": report.model_dump(),
            "summary": summary,
            "log": logger.path.read_text(encoding="utf-8"),
            "log_path": str(logger.path),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050)
