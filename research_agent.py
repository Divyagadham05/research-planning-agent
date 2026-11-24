"""
Research & Planning Agent (LangGraph) - agent logic (no UI). Used by streamlit_app.py.
Flow: planner -> human approval (editable plan) -> parallel research (Send) -> writer -> reviewer loop -> finalize.
Needs the environment variables GROQ_API_KEY and TAVILY_API_KEY (Streamlit secrets are copied to them).
"""
import os, tempfile, uuid

def get_key(name):
    key = (os.environ.get(name) or "").strip()
    if not key:
        raise RuntimeError(f"{name} is missing. Add it in the app's Secrets settings.")
    if not key.isascii():
        raise RuntimeError(f"{name} contains non-English characters. Re-copy the key without extra text.")
    return key

os.environ["GROQ_API_KEY"] = get_key("GROQ_API_KEY")
TAVILY_API_KEY = get_key("TAVILY_API_KEY")

# ---------------- Imports and configuration ----------------

import re, json, time, uuid, operator
from datetime import date
from typing import TypedDict, Annotated, List
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from tavily import TavilyClient
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

# ---------------- Configuration (change here, nothing else needs editing) ----------------
MODEL_NAME = "qwen/qwen3.8-27b"      # the model that worked in your Groq account.
                                     # To change it, pick another id from the "Chat models available" list above.
MODEL_PROVIDER = "groq"              # any provider supported by init_chat_model
DEFAULT_MAX_ITERATIONS = 2           # review rounds (hard limit)

# --- Token budget (IMPORTANT for the Groq free tier) ---
# Your qwen model allows only 1000 OUTPUT tokens per minute, so each call must stay below that.
# On a model/tier with higher limits you can raise MAX_OUTPUT_TOKENS (e.g. 3000) and REPORT_WORDS (e.g. 900).
MAX_OUTPUT_TOKENS = 900
REPORT_WORDS = 450                   # the writer is told to stay under this length
RESULTS_PER_QUERY = 3
MAX_CHARS_PER_SOURCE = 700
BLOCKED_DOMAINS = [
    "facebook.com", "instagram.com", "twitter.com", "x.com", "tiktok.com",
    "reddit.com", "quora.com", "pinterest.com", "linkedin.com", "youtube.com",
]

# Qwen 3.8 is a "thinking" model by default. Its hidden reasoning tokens COUNT as output tokens and used up
# your 1000 tokens/minute limit. reasoning_effort="none" switches it to instruct mode (no thinking).
llm_kwargs = dict(temperature=0.7, max_tokens=MAX_OUTPUT_TOKENS)     # 0.7 = Groq's recommended instruct setting
if MODEL_NAME.startswith("qwen/"):
    try:
        llm = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER, reasoning_effort="none", **llm_kwargs)
    except Exception:
        llm = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER,
                              model_kwargs={"reasoning_effort": "none"}, **llm_kwargs)
else:
    llm = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER, **llm_kwargs)
tavily = TavilyClient(api_key=TAVILY_API_KEY)
print("Model and search client ready.")

# ---------------- Schemas and state ----------------

# ---------------- Structured output schemas ----------------
class PlanTask(BaseModel):
    task_id: int
    title: str
    query: str = Field(description="Short web search query, keywords only, max 10 words")
    description: str = Field(description="One sentence: what this task should find out")

class ResearchPlan(BaseModel):
    tasks: List[PlanTask]

class ReviewResult(BaseModel):
    approved: bool = Field(description="True only if every claim is supported by its cited source, citations are valid, and no important gap remains")
    unsupported_claims: List[str] = Field(default_factory=list, description="Claims not supported by the cited source")
    gaps: List[str] = Field(default_factory=list, description="Important missing topics")
    follow_up_queries: List[str] = Field(default_factory=list, description="Up to 3 short web search queries that would fill the gaps; empty if none needed")

class ActionStep(BaseModel):
    step: int
    action: str
    priority: str = Field(description="High, Medium or Low")
    timeline: str = Field(description="For example: Week 1-2")
    sources: List[int] = Field(default_factory=list, description="Supporting source numbers; empty if general advice")

class ActionPlan(BaseModel):
    steps: List[ActionStep]

# ---------------- Graph state ----------------
class AgentState(TypedDict, total=False):
    question: str
    max_iterations: int
    plan: list
    approval_status: str
    task: dict                                          # input for one parallel worker
    research_results: Annotated[list, operator.add]     # parallel workers append here
    sources: list
    report: str
    review: dict
    iteration: int
    final_report: str

# ---------------- Helper functions ----------------

# ---------------- Helper functions ----------------
def clean(text: str) -> str:
    # Remove hidden reasoning tags some models print
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()

def with_retry(fn, tries=4, wait=2):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "Request too large" in msg:
                raise RuntimeError("Request exceeds your Groq limit. Lower MAX_OUTPUT_TOKENS / REPORT_WORDS "
                                   "or choose another MODEL_NAME. Qwen must run with reasoning effort set to none. Details: " + msg[:300])
            if i == tries - 1:
                raise
            if "429" in msg or "rate limit" in msg.lower():
                print("Groq per-minute limit reached. Waiting 65 seconds...")
                time.sleep(65)                    # per-minute quota resets
            else:
                time.sleep(wait * (i + 1))

def ask_llm(system, human):
    resp = with_retry(lambda: llm.invoke([("system", system), ("human", human)]))
    return clean(resp.content)

def ask_structured(schema, system, human):
    structured = llm.with_structured_output(schema)
    return with_retry(lambda: structured.invoke([("system", system), ("human", human)]))

def is_blocked(url):
    host = urlparse(url).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS)

def search_web(query):
    """Tavily search with retry. Never raises: returns (findings, error_message)."""
    try:
        resp = with_retry(lambda: tavily.search(
            query=query, search_depth="basic",
            max_results=RESULTS_PER_QUERY + 2, exclude_domains=BLOCKED_DOMAINS))
    except Exception as e:
        return [], f"Search failed: {e}"
    findings = []
    for r in resp.get("results", []):
        url = str(r.get("url", "")).strip()
        if not url.startswith("http") or is_blocked(url):
            continue
        findings.append({
            "title": (r.get("title") or url).replace("[", "(").replace("]", ")"),
            "url": url,
            "content": (r.get("content") or "")[:MAX_CHARS_PER_SOURCE],
        })
        if len(findings) >= RESULTS_PER_QUERY:
            break
    return findings, ""

def build_sources(results):
    """De-duplicate by URL and number sources 1..N (stable order by task id)."""
    sources, seen = [], set()
    for r in sorted(results, key=lambda x: x["task_id"]):
        for f in r["findings"]:
            if f["url"] in seen:
                continue
            seen.add(f["url"])
            sources.append({"id": len(sources) + 1, "title": f["title"], "url": f["url"],
                            "content": f["content"], "task": r["title"]})
    return sources

def evidence_text(sources):
    return "\n\n".join(
        f"[{s['id']}] {s['title']}\nURL: {s['url']}\n(Research task: {s['task']})\nEvidence: {s['content']}"
        for s in sources)

def check_citations(report, n_sources):
    """Code-based citation check: invalid numbers, uncited paragraphs, unused sources."""
    cited = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", report):
        cited.update(int(x) for x in re.split(r"\s*,\s*", group))
    invalid = sorted(c for c in cited if c < 1 or c > n_sources)
    uncited = []
    for para in re.split(r"\n\s*\n", report):
        p = para.strip()
        if len(p) > 200 and not p.startswith("#") and not re.search(r"\[\d+", p):
            uncited.append(p[:80] + "...")
    unused = sorted(set(range(1, n_sources + 1)) - cited)
    return {"invalid": invalid, "uncited": uncited, "unused": unused}

def research_metrics(results):
    """Measured parallel time vs. the sum of each search (= sequential estimate), first round only."""
    first = [r for r in results if r["task_id"] < 100]
    if not first:
        return {}
    wall = max(r["ended"] for r in first) - min(r["started"] for r in first)
    seq = sum(r["ended"] - r["started"] for r in first)
    return {"parallel_seconds": round(wall, 2),
            "sequential_estimate_seconds": round(seq, 2),
            "speedup": round(seq / wall, 2) if wall > 0 else None}

# ---------------- Graph nodes ----------------

# ---------------- Graph nodes ----------------
def planner_node(state: AgentState):
    plan = ask_structured(
        ResearchPlan,
        f"Today's date is {date.today():%B %d, %Y}. "
        "You are a research planning assistant. Break the user's request into exactly 3 independent "
        "research tasks that cover different angles. For each task give a short title, a concise web search "
        "query (keywords only, max 10 words) and a one-sentence description. "
        "Never put an outdated year in a query: use the current year or no year at all.",
        state["question"])
    tasks = [{"task_id": i + 1, "title": t.title, "query": t.query, "description": t.description}
             for i, t in enumerate(plan.tasks[:5])]
    return {"plan": tasks, "iteration": 0}

def human_approval_node(state: AgentState):
    # Pauses the graph. Resume with {"decision": "approved"|"rejected", "plan": [...optional edited plan...]}
    reply = interrupt({
        "message": "Review the research plan. You may edit it, approve it, or reject it.",
        "question": state["question"],
        "research_plan": state["plan"],
    })
    if isinstance(reply, dict):
        decision, edited = str(reply.get("decision", "approved")).lower(), reply.get("plan")
    else:
        decision, edited = str(reply).lower(), None
    if decision not in ("approved", "approve", "yes"):
        return {"approval_status": "rejected"}
    update = {"approval_status": "approved"}
    if edited:
        update["plan"] = edited
    return update

def route_after_approval(state: AgentState):
    if state.get("approval_status") != "approved":
        return END
    return [Send("research_worker", {"task": t}) for t in state["plan"]]   # parallel fan-out

def research_worker(state):
    task = state["task"]
    started = time.time()
    findings, error = search_web(task["query"])
    return {"research_results": [{
        "task_id": task["task_id"], "title": task["title"], "query": task["query"],
        "findings": findings, "error": error, "started": started, "ended": time.time()}]}

WRITER_SYSTEM = f"""You are an evidence-based research writer. Use ONLY the numbered sources supplied.

Rules:
- Every factual claim, number or statistic must end with a citation such as [2] or [1][3]. Only use source numbers that exist.
- Never add facts from memory. If the sources do not cover something, write: "Evidence not found in the collected sources."
- Attribute numbers to their source ("According to [2], ..."). Do not present vendor or blog claims as proven universal facts, and mention when a source is a blog or vendor page.
- Do NOT write a References section and do NOT write an Action Plan (both are added automatically).
- Use these markdown sections: ## Executive Summary, ## Key Findings (one ### sub-heading per research task; follow-up searches are supplementary), ## Benefits, ## Risks and Limitations, ## Conclusion.
- Keep the whole report under {REPORT_WORDS} words (be concise) and finish every section completely."""

def report_writer_node(state: AgentState):
    sources = build_sources(state.get("research_results", []))
    if not sources:
        return {"sources": [], "report": "Evidence not found: all searches failed or returned no usable sources."}
    human = f"Research question: {state['question']}\n\nSources:\n{evidence_text(sources)}"
    review = state.get("review")
    if review and not review.get("approved"):
        human += ("\n\nA reviewer rejected the previous draft. Fix these problems in the new report:\n"
                  f"Unsupported claims: {review.get('unsupported_claims')}\n"
                  f"Gaps: {review.get('gaps')}\n"
                  f"Citation problems: {review.get('citation_check')}")
    return {"sources": sources, "report": ask_llm(WRITER_SYSTEM, human)}

REVIEW_SYSTEM = """You are a strict fact-checking reviewer. Compare the report with the numbered sources.
List claims that the cited source does not support (unsupported_claims), important missing topics (gaps),
and up to 3 short web search queries that would fill the gaps (follow_up_queries, empty if none are needed).
Set approved=true ONLY if all claims are supported, citations are valid, every section is complete and no important gap remains. A source that is retrieved but not cited is NOT a gap, and minor style points must not prevent approval."""

def reviewer_node(state: AgentState):
    iteration = state.get("iteration", 0) + 1
    sources, report = state.get("sources", []), state["report"]
    if not sources:   # nothing to verify or improve
        return {"iteration": iteration, "review": {
            "approved": True, "unsupported_claims": [], "gaps": ["No sources could be retrieved."],
            "follow_up_queries": [], "citation_check": {}}}
    check = check_citations(report, len(sources))
    result = ask_structured(
        ReviewResult, REVIEW_SYSTEM,
        f"Question: {state['question']}\n\nSources:\n{evidence_text(sources)}\n\nReport:\n{report}\n\n"
        f"Automatic citation check: {check}")
    approved = result.approved and not check["invalid"] and not check["uncited"]
    return {"iteration": iteration, "review": {
        "approved": approved,
        "unsupported_claims": result.unsupported_claims,
        "gaps": result.gaps,
        "follow_up_queries": [q.strip() for q in result.follow_up_queries if q.strip()][:3],
        "citation_check": check}}

def route_after_review(state: AgentState):
    review = state["review"]
    if review["approved"] or state["iteration"] >= state.get("max_iterations", DEFAULT_MAX_ITERATIONS):
        return "finalize"                                         # done (or limit reached)
    queries = review.get("follow_up_queries") or []
    if queries:                                                   # re-search to fill gaps
        base = 100 * state["iteration"]
        return [Send("research_worker", {"task": {
                    "task_id": base + i + 1, "title": "Follow-up: " + q, "query": q}})
                for i, q in enumerate(queries)]
    return "report_writer"                                        # rewrite only

ACTION_SYSTEM = """You are a planning expert. Based on the research report, create 4 to 6 concrete, ordered steps
that help the user achieve their goal. For each step give: the action, a priority (High, Medium or Low),
a realistic timeline (for example "Week 1-2") and the source numbers that support it (empty list if it is general advice)."""

def finalize_node(state: AgentState):
    question, report = state["question"], state["report"]
    sources, review = state.get("sources", []), state.get("review", {})
    out = [f"# Research Report: {question}", "", report, ""]

    if sources:
        plan = ask_structured(ActionPlan, ACTION_SYSTEM, f"Goal: {question}\n\nReport:\n{report}")
        out += ["## Action Plan", "", "| # | Action | Priority | Timeline | Sources |", "|---|---|---|---|---|"]
        for s in plan.steps:
            cites = ", ".join(f"[{n}]" for n in s.sources if 1 <= n <= len(sources)) or "General"
            action = s.action.replace("|", "/").replace("\n", " ")
            out.append(f"| {s.step} | {action} | {s.priority} | {s.timeline} | {cites} |")
        out.append("")

    unused = set(check_citations(report, len(sources))["unused"]) if sources else set()
    out += ["## References", ""]
    if not sources:
        out.append("No sources were retrieved.")
    out += [f"{s['id']}. [{s['title']}]({s['url']})" for s in sources if s["id"] not in unused]
    if unused:
        out += ["", "**Also retrieved (not cited in the report):**", ""]
        out += [f"{s['id']}. [{s['title']}]({s['url']})" for s in sources if s["id"] in unused]

    notes = [f"Review rounds used: {state.get('iteration', 0)}.",
             f"Reviewer approved: {'yes' if review.get('approved') else 'no'}."]
    if not review.get("approved"):
        if review.get("unsupported_claims"):
            notes.append("Unresolved unsupported claims: " + "; ".join(review["unsupported_claims"]))
        if review.get("gaps"):
            notes.append("Unresolved gaps: " + "; ".join(review["gaps"]))
    out += ["", "## Quality Check", ""] + [f"- {n}" for n in notes]
    return {"final_report": "\n".join(out)}

# ---------------- Build the graph ----------------

# ---------------- Build and compile the graph ----------------
builder = StateGraph(AgentState)
builder.add_node("planner", planner_node)
builder.add_node("human_approval", human_approval_node)
builder.add_node("research_worker", research_worker)
builder.add_node("report_writer", report_writer_node)
builder.add_node("reviewer", reviewer_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "human_approval")
builder.add_conditional_edges("human_approval", route_after_approval, ["research_worker", END])
builder.add_edge("research_worker", "report_writer")
builder.add_edge("report_writer", "reviewer")
builder.add_conditional_edges("reviewer", route_after_review, ["research_worker", "report_writer", "finalize"])
builder.add_edge("finalize", END)

memory = MemorySaver()   # development. For production use SqliteSaver / PostgresSaver
agent = builder.compile(checkpointer=memory)

# ---------------- Run helpers ----------------

# ---------------- Run helpers (used by the test cell and the UI) ----------------
def start_run(question, max_iterations=DEFAULT_MAX_ITERATIONS):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = agent.invoke({"question": question, "max_iterations": int(max_iterations),
                           "iteration": 0, "research_results": [], "sources": []}, config)
    interrupts = result.get("__interrupt__")
    plan = interrupts[0].value["research_plan"] if interrupts else []
    return config, plan

def resume_run(config, decision, plan=None):
    payload = {"decision": decision}
    if plan:
        payload["plan"] = plan
    return agent.invoke(Command(resume=payload), config)

def parse_plan(text):
    """Validate the (possibly edited) plan JSON from the UI."""
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("tasks", [])
    if not isinstance(data, list) or not 1 <= len(data) <= 6:
        raise ValueError("Plan must be a list of 1 to 6 tasks.")
    tasks = []
    for i, t in enumerate(data, start=1):
        t = dict(t)
        t["task_id"] = i                              # keep ids below 100 (100+ = follow-ups)
        t.setdefault("description", t.get("title", ""))
        tasks.append(PlanTask(**t).model_dump())
    return tasks
