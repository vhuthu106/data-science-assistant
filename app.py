import streamlit as st
from tools.dataset_tools import DatasetManager
from agents.planner import Planner
from agents.worker import Worker
from core.run_log import RunLog, run_status
from core.llm import get_client, get_model, LLMUnavailable, MissingAPIKeyError
from pathlib import Path
import pandas as pd
import time

# Paths are built from THIS file's location, so the app works no matter
# which folder you start it from.
BASE_DIR = Path(__file__).resolve().parent
TITANIC_PATH = BASE_DIR / "datasets" / "Titanic-Dataset.csv"

st.set_page_config(page_title="Data Science Assistant", layout="wide")
st.title("Data Science Assistant")

# Fail fast with a clear message instead of crashing on the first question
try:
    get_client()
except MissingAPIKeyError as e:
    st.error(f"🔑 {e}")
    st.stop()


# Streamlit re-runs this whole script on EVERY click or keypress.
# @st.cache_resource builds these objects once and reuses them on every rerun.
@st.cache_resource
def load_agents():
    return Planner(), Worker(), RunLog()

planner, worker, run_log = load_agents()


def show_value(value):
    """Display a step's `result` in the most readable form."""
    if value is None:
        return
    if isinstance(value, (pd.DataFrame, pd.Series)):
        st.dataframe(value)
    elif type(value).__module__.startswith("matplotlib"):
        return  # a chart object: the chart itself is already shown as an image
    else:
        st.write(value)

# st.session_state survives reruns (per browser session), so the loaded
# dataset is still there after the next click.
if "dm" not in st.session_state:
    st.session_state.dm = DatasetManager()
dm = st.session_state.dm

# Dataset management section
st.sidebar.header("📁 Dataset Management")

# Option 1: Load from predefined datasets
st.sidebar.subheader("Load Example Dataset")
if st.sidebar.button("Load Titanic Dataset"):
    if not TITANIC_PATH.exists():
        st.sidebar.error(f"Titanic dataset not found at {TITANIC_PATH}")
    elif dm.load_from_path("titanic", TITANIC_PATH):
        st.sidebar.success("✓ Titanic dataset loaded!")

# Option 2: Upload custom dataset
st.sidebar.subheader("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload a CSV file for analysis"
)

# The uploader returns the same file on every rerun, so only load it when it is
# a NEW upload. Otherwise it would silently replace a dataset chosen afterwards.
if uploaded_file is not None and st.session_state.get("upload_id") != uploaded_file.file_id:
    if dm.load_from_upload(uploaded_file):
        st.session_state.upload_id = uploaded_file.file_id
        st.sidebar.success(f"✓ {uploaded_file.name} loaded successfully!")

# Display dataset info
st.sidebar.subheader("Dataset Info")
if dm.get_dataset() is not None:
    dataset_info = dm.get_dataset_info()
    st.sidebar.write(f"**Name:** {dataset_info['name']}")
    st.sidebar.write(f"**Shape:** {dataset_info['shape']}")
    st.sidebar.write(f"**Columns:** {', '.join(dataset_info['columns'])}")
else:
    st.sidebar.info("No dataset loaded. Please load or upload a dataset.")

# Display which model is in use
st.sidebar.subheader("Model")
st.sidebar.info(f"🤖 {get_model()}")

# Instructions
st.sidebar.header("ℹ️ Instructions")
st.sidebar.info("""
1. Load a dataset using the sidebar options
2. Ask a data science question in natural language
3. The assistant will create and execute a plan
4. Results will include plots, tables, and code
""")

st.sidebar.header("💡 Example Queries")
st.sidebar.text("""
- Plot survival rate by Sex
- Show average age by class
- Analyze correlation between features
- Create histogram of ages
- Show missing values
- Group by embarked and show stats
""")

# --- Tabs for navigation ---
tab1, tab2 = st.tabs(["🔍 Data Analysis", "📊 Run Metrics"])

# ---------------------------
# TAB 1: Data Analysis
# ---------------------------
with tab1:

    if dm.get_dataset() is None:
        st.warning("Please load a dataset first using the sidebar options.")
    else:
        st.subheader("📊 Dataset Preview")
        st.dataframe(dm.get_dataset().head())

        user_query = st.text_input(
            "Ask a data science question:",
            placeholder="e.g., 'Plot survival rate by Sex', 'Show average age by class'",
            key="query_input"
        )

        if st.button("Run Analysis", type="primary") and user_query.strip():
            start_time = time.time()  # Start timing
            
            with st.spinner("Planning and executing..."):
                try:
                    # Create a validated plan (column names and types are sent, never rows)
                    plan = planner.plan(user_query, dm.get_dataset())

                    st.subheader("📋 Analysis Plan")
                    if plan.get("fallback_reason"):
                        st.warning(f"The model's plan was unusable ({plan['fallback_reason']}), "
                                   "so a simple rule-based plan is used instead.")
                    st.json(plan)

                    # Execute the plan, step by step
                    result = worker.handle(plan, dm)

                    # Log what ACTUALLY happened (success / partial / failed), not just "no crash"
                    run_log.record(user_query, get_model(), seconds=time.time() - start_time,
                                   **run_status(result))

                    st.subheader("📈 Results:")
                    if result.get("stopped"):
                        st.error(f"⛔ Stopped early: {result['stopped']}")
                    for number, step in enumerate(result.get("steps", []), start=1):
                        st.markdown(f"**Step {number} · `{step['action']}`** — {step['instruction']}")
                        if step["skipped"]:
                            st.info("Skipped because an earlier step did not complete.")
                            continue
                        repairs = len(step["failed_attempts"])
                        if step["success"]:
                            if repairs:
                                st.caption(f"🔧 Self-repaired: {repairs} failed attempt(s) before the model's code worked "
                                           "(details under 'Code and printed output').")
                            for png in step["figures"]:
                                st.image(png)
                            show_value(step["result"])
                        else:
                            st.error(f"❌ {step['error']}")
                        for warning in step["warnings"]:
                            st.warning(f"⚠️ {warning}")
                        if step["code"]:
                            with st.expander("Code and printed output"):
                                st.code(step["code"], language="python")
                                if step["output"]:
                                    st.text(step["output"])
                                for i, failed in enumerate(step["failed_attempts"], start=1):
                                    st.markdown(f"**Failed attempt {i}:** `{failed['error']}`")
                                    st.code(failed["code"], language="python")

                except LLMUnavailable as e:
                    # The model provider could not answer (rate limit, overload, timeout, bad key...)
                    run_log.record(user_query, get_model(), "provider_error", time.time() - start_time, error=str(e))
                    st.error(f"🌐 {e}")

                except Exception as e:
                    run_log.record(user_query, get_model(), "error", time.time() - start_time, error=repr(e))
                    st.error(f"Error during analysis: {e}")

# ---------------------------
# TAB 2: Run Metrics
# ---------------------------
with tab2:
    st.header("📊 Run Metrics")
    st.caption("One row per question asked. Shows how often the assistant fully answers, "
               "and how often self-repair rescues code that failed the first time.")
    runs = run_log.load()
    if runs.empty:
        st.info("No runs recorded yet. Ask a question in the Data Analysis tab.")
    else:
        s = RunLog.summary(runs)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Questions", s["runs"])
        c2.metric("Fully answered", f"{s['success_rate']:.0%}")
        c3.metric("Median time", f"{s['median_seconds']:.1f}s")
        c4.metric("Self-repair rescued",
                  "n/a" if s["repair_success_rate"] is None else f"{s['repair_success_rate']:.0%}",
                  help="Of the steps whose first code failed, the share the model then fixed.")
        st.write(f"Partial or failed answers: **{s['partial_or_failed']}** · "
                 f"Provider errors (rate limit, overload...): **{s['provider_errors']}**")

        st.subheader("Outcomes")
        st.bar_chart(runs["status"].value_counts())

        st.subheader("Recent questions")
        st.dataframe(runs[["timestamp", "question", "status", "steps_ok", "steps_total",
                           "failed_attempts", "seconds", "model", "error"]].head(20),
                     hide_index=True)
