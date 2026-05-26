# Nexus AI — Multi-Agent Research Assistant

> A production-grade, multi-agent research assistant built with **LangGraph**, **FastAPI**, and a modern **Glassmorphism UI**. The system orchestrates four specialized agents that collaborate to research businesses, validate findings, and deliver coherent summaries — all while supporting multi-turn conversations and human-in-the-loop clarification.

---

## Table of Contents
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Agent Workflow](#agent-workflow)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Increment Log](#increment-log)

---

## Architecture

The application follows a **modular, layered architecture** with clear separation of concerns:

| Layer | Responsibility | Location |
|-------|---------------|----------|
| **Agent Layer** | LangGraph state machine, agent logic, routing | `src/agent/` |
| **API Layer** | FastAPI endpoints, request/response handling | `src/api/` |
| **Presentation Layer** | Frontend UI (HTML/CSS/JS) | `src/web/` |

---

## Directory Structure

```text
nexus-ai-research-assistant/
├── src/
│   ├── __init__.py
│   ├── core/                      # App-wide configuration
│   │   └── config.py              # Centralized settings (models, thresholds, API keys)
│   ├── agent/                     # LangGraph multi-agent system
│   │   ├── __init__.py
│   │   ├── state.py               # AgentState schema (TypedDict)
│   │   ├── graph.py               # Routing logic & graph compilation
│   │   ├── core/                  # Agent utilities (split by domain)
│   │   │   ├── llm.py             # LLM factory, retry logic, JSON extraction
│   │   │   ├── memory.py          # Sliding window memory management
│   │   │   └── tools.py           # Tavily search & text truncation
│   │   └── nodes/                 # One file per agent (SRP)
│   │       ├── __init__.py
│   │       ├── clarity.py         # Clarity Agent & clarification handler
│   │       ├── research.py        # Research Agent (query building & search)
│   │       ├── validator.py       # Validator Agent (quality assessment)
│   │       └── synthesis.py       # Synthesis Agent (final markdown output)
│   ├── api/                       # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py                # App instantiation, middleware, static files
│   │   ├── routes.py              # API endpoint handlers (APIRouter)
│   │   └── schemas.py             # Pydantic request/response models
│   └── web/                       # Frontend application
│       ├── index.html             # Semantic HTML with agent pipeline tracker
│       ├── styles.css             # Glassmorphism dark theme
│       └── app.js                 # Client-side logic, pipeline animation
├── .env                           # Environment variables (API keys)
├── .env.example                   # Template for environment setup
├── requirements.txt               # Python dependencies
├── ai_prompts_and_reasoning.tex   # AI prompts & reasoning document (LaTeX)
├── testing.txt                    # Full stress test transcript
└── README.md                      # Project documentation
```

---

## Agent Workflow

```
User Query
    │
    ▼
┌──────────────┐
│ Clarity Agent │──── needs_clarification ──► ASK USER (Human-in-the-Loop)
└──────┬───────┘
       │ clear
       ▼
┌────────────────┐
│ Research Agent  │──── confidence ≥ 6 ──────────────────┐
└──────┬─────────┘                                       │
       │ confidence < 6                                  │
       ▼                                                 │
┌──────────────────┐                                     │
│ Validator Agent   │──── sufficient ────────────────────►│
└──────┬───────────┘                                     │
       │ insufficient                                    │
       │ (attempts < 3)                                  ▼
       └──── loop back to Research Agent ──►    ┌─────────────────┐
                                                │ Synthesis Agent  │
                                                └────────┬────────┘
                                                         │
                                                         ▼
                                                    Final Answer
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph + LangChain |
| LLMs (Multi-Model) | Groq (`llama-3.1-8b-instant` for fast tasks, `llama-3.3-70b-versatile` for synthesis) |
| Search Tool | Tavily Search API (Optimized with `include_answer=True`) |
| Backend | FastAPI + Pydantic (with SSE streaming) |
| Frontend | Vanilla HTML/CSS/JS (Glassmorphism, Real-time Pipeline UI) |
| State Persistence | LangGraph MemorySaver |
| Quality Assurance | Pytest (23 unit tests) |
| Deployment | Docker |

---

## Design Patterns & Engineering Practices

The codebase was explicitly refactored to adhere to industry-grade standards:

*   **Layered Architecture:** Strict separation between Presentation (`src/web`), Application (`src/api`), and Domain (`src/agent`) layers.
*   **Low Coupling / High Cohesion:** 
    *   `src/agent/core/` splits utilities by domain: `llm.py` (LLM factory), `memory.py` (sliding window), `tools.py` (Tavily search).
    *   `src/core/config.py` acts as a single source of truth for all tunable parameters.
*   **Factory Pattern:** `get_llm(purpose)` creates the appropriate LLM instance based on task complexity.
*   **Strategy Pattern:** Different models are applied dynamically (fast models for routing, powerful models for synthesis).
*   **Single Responsibility Principle:** Each agent lives in its own file under `src/agent/nodes/`.
*   **Resilience & Fallbacks:** Automatic retries with exponential backoff on rate limits, and model fallback mechanisms.
*   **Sliding Window Memory:** Keeps recent messages verbatim and summarizes older history to prevent token overflow.
*   **Real-time Feedback (SSE):** The UI uses Server-Sent Events to stream agent activations in real-time, drastically improving UX by removing "black box" waiting periods.
*   **Structured Logging:** Replaced all print statements with a dedicated `logging.py` module for production observability.

---

## Stress Testing & Quality Assurance

The system underwent **two rounds of rigorous stress testing** across 10+ test scenarios. Each round was designed to expose specific failure modes:

| Test Category | Scenarios | Bugs Found & Fixed |
|---------------|-----------|--------------------|
| **Multi-Turn Memory** | Pronoun resolution ("their CEO"), topic switching (Microsoft → Apple) | Clarity Agent failed on topic switches; fixed via principle-based prompt engineering |
| **State Persistence** | Follow-up queries after topic change | Stale `research_data` from prior topics polluted new searches; fixed by resetting state in Clarity Agent |
| **Token Limits** | Large Tavily results, long conversations | Groq 413 error (6,000 TPM exceeded); fixed via `truncate_text()` and sliding window memory |
| **Edge Cases** | Empty input, gibberish ("skibidi toilet rizz ohio"), obscure entities | Clarity Agent correctly rejects nonsense; system handles gracefully |
| **Validator Loop** | Queries with no data (Acme Widget 1892) | Validator correctly identifies insufficient data; system loops then gives honest response |
| **Massive Output** | WW2 comprehensive overview | No crashes; truncation prevents token overflow on Groq free tier |

Full test transcript available in `testing.txt`.

---

## Setup & Installation

### 1. Clone & enter the project
```bash
cd nexus-ai-research-assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the environment
```powershell
# Windows (PowerShell)
.\venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API keys
Copy `.env.example` to `.env` and fill in your keys:
```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## Running the Application

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Then open **http://localhost:8000** in your browser.

---

## Increment Log

| Increment | Description | Status |
|-----------|-------------|--------|
| 1 | Project scaffolding, modular directory structure, `__init__.py` packages | ✅ Complete |
| 2 | Core agent logic: Clarity, Research, Validator, Synthesis nodes | ✅ Complete |
| 3 | LangGraph state machine with conditional routing & MemorySaver | ✅ Complete |
| 4 | FastAPI backend with `/chat` endpoint & CORS middleware | ✅ Complete |
| 5 | Glassmorphism frontend UI with Markdown rendering & typing indicator | ✅ Complete |
| 6 | Bug fix: Updated Groq model from decommissioned `llama3-8b-8192` to `llama-3.1-8b-instant` | ✅ Complete |
| 7 | Testing, debugging & edge-case hardening (Token limits, conversation context) | ✅ Complete |
| 8 | Refactor: Design patterns (Factory, Strategy), module extraction (`core/`, `nodes/`) | ✅ Complete |
| 9 | UX Polish: Implemented Server-Sent Events (SSE) for real-time UI agent tracking | ✅ Complete |
| 10 | Enterprise Readiness: Structured logging, 23 Pytest unit tests, Dockerfile | ✅ Complete |
| 11 | Documentation: Code comments, docstrings, `ai_prompts_and_reasoning.tex` | ✅ Complete |
| 12 | Final polish & packaging | ✅ Complete |
