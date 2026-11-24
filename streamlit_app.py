import os, json
import streamlit as st

st.set_page_config(page_title="Research & Planning Agent", page_icon="🔎", layout="wide")

# Streamlit secrets -> environment variables (the agent reads os.environ)
for _k in ("GROQ_API_KEY", "TAVILY_API_KEY", "APP_PASSWORD"):
    try:
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

ss = st.session_state

# Optional password gate: set APP_PASSWORD in the app's secrets to turn it on
_pw = os.environ.get("APP_PASSWORD")
if _pw and not ss.get("auth_ok"):
    typed = st.text_input("Password", type="password")
    if typed and typed == _pw:
        ss.auth_ok = True
        st.rerun()
    st.info("Enter the password to continue.")
    st.stop()

try:
    import research_agent as ra          # builds the LangGraph agent once per server process
except Exception as e:
    st.error(f"Setup problem: {e}")
    st.stop()

ss.setdefault("stage", "input")          # input -> plan -> done

st.title("Research & Planning Agent")
st.caption("LangGraph agent: plans, asks for your approval, searches the web in parallel (Tavily), writes a cited report (Qwen on Groq), reviews itself, and adds an action plan.")

if ss.get("notice"):
    st.info(ss.pop("notice"))

question = st.text_area("Research / planning request", key="question", height=100,
                        placeholder="e.g. Plan a 3-month study roadmap to learn data engineering")
max_iter = st.slider("Max review rounds", 1, 4, 2)

if st.button("Step 1: Generate plan", type="primary"):
    if not question.strip():
        st.warning("Please enter a research question.")
    else:
        try:
            with st.spinner("Planning..."):
                cfg, plan = ra.start_run(question.strip(), max_iter)
            ss.config, ss.plan_text, ss.stage = cfg, json.dumps(plan, indent=2), "plan"
            ss.pop("result", None)
        except Exception as e:
            st.error(f"Error while planning: {e}")

if ss.stage == "plan":
    st.subheader("Research plan (editable JSON)")
    st.text_area("Edit the plan if you want, then approve or reject it.", key="plan_text", height=320)
    c1, c2 = st.columns(2)
    if c1.button("Step 2: Approve and run", type="primary"):
        try:
            plan = ra.parse_plan(ss.plan_text)
        except Exception as e:
            st.error(f"Plan JSON problem: {e}")
        else:
            finished = False
            try:
                with st.spinner("Researching and writing... this can take several minutes on the free Groq tier."):
                    ss.result = ra.resume_run(ss.config, "approved", plan)
                ss.stage = "done"
                finished = True
            except Exception as e:
                st.error(f"Error during research: {e}")
            if finished:
                st.rerun()
    if c2.button("Reject plan"):
        rejected = False
        try:
            ra.resume_run(ss.config, "rejected")
            ss.stage, ss.notice = "input", "Plan rejected. Nothing was researched."
            rejected = True
        except Exception as e:
            st.error(f"Error: {e}")
        if rejected:
            st.rerun()

if ss.stage == "done" and ss.get("result"):
    res = ss.result
    review = res.get("review", {})
    st.success(f"Done. Review rounds: {res.get('iteration')}. Reviewer approved: {review.get('approved')}. "
               f"Sources: {len(res.get('sources', []))}.")
    report = res.get("final_report", "")
    st.markdown(report)
    st.download_button("Download report (.md)", report, file_name="research_report.md", mime="text/markdown")
    with st.expander("Parallel research timing"):
        st.json(ra.research_metrics(res.get("research_results", [])))
    if st.button("Start a new question"):
        ss.stage = "input"
        ss.pop("result", None)
        st.rerun()
