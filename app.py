import streamlit as st
from tools.dataset_tools import DatasetManager
from agents.planner import Planner
from agents.worker import Worker
from core.entropy_logger import EntropyLogger  # Added
from core.llm import get_client, get_model, MissingAPIKeyError
import os
from pathlib import Path
import pandas as pd
import sqlite3
import time  # Added for execution time tracking

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
    return Planner(), Worker(), EntropyLogger()

planner, worker, logger = load_agents()


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

# Entropy Metrics in Sidebar
st.sidebar.header("📈 Entropy Metrics")
if st.sidebar.button("Show Entropy Report"):
    try:
        metrics = logger.get_entropy_metrics()
        
        if "error" in metrics:
            st.sidebar.error(metrics["error"])
        else:
            st.sidebar.success("✅ Entropy Tracking Active")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Total Queries", metrics['total_interactions'])
                st.metric("Success Rate", f"{metrics['success_rate']*100:.1f}%")
            with col2:
                st.metric("Avg Time", f"{metrics['average_fix_time']:.2f}s")
                st.metric("Total Errors", metrics['total_errors'])
    except Exception as e:
        st.sidebar.error(f"Could not load entropy metrics: {e}")

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
tab1, tab2 = st.tabs(["🔍 Data Analysis", "📊 Entropy Metrics"])

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

                    execution_time = time.time() - start_time  # Calculate execution time
                    
                    # Log successful execution
                    logger.log_interaction(
                        prompt_text=user_query, 
                        response_text="Analysis completed successfully", 
                        error_count=0, 
                        fix_time=execution_time
                    )

                    st.subheader("📈 Results:")
                    for number, step in enumerate(result.get("steps", []), start=1):
                        st.markdown(f"**Step {number} · `{step['action']}`** — {step['instruction']}")
                        if step["success"]:
                            for png in step["figures"]:
                                st.image(png)
                            show_value(step["result"])
                        else:
                            st.error(f"❌ {step['error']}")
                        for warning in step["warnings"]:
                            st.warning(f"⚠️ {warning}")
                        with st.expander("Code and printed output"):
                            st.code(step["code"], language="python")
                            if step["output"]:
                                st.text(step["output"])

                except Exception as e:
                    execution_time = time.time() - start_time  # Calculate execution time even for errors
                    
                    # Log error with execution time
                    logger.log_interaction(
                        prompt_text=user_query, 
                        response_text=f"Error: {str(e)}", 
                        error_count=1, 
                        fix_time=execution_time
                    )
                    
                    st.error(f"Error during analysis: {str(e)}")

# ---------------------------
# TAB 2: Entropy Metrics
# ---------------------------
with tab2:
    st.header("📊 Entropy Metrics Dashboard")

    if os.path.exists("entropy_log.db"):
        try:
            # Use the logger's get_logs method which includes compatibility columns
            df = logger.get_logs(limit=1000)  # Get all logs
            
            if not df.empty:
                st.subheader("📈 System Performance Overview")
                
                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_interactions = len(df)
                    st.metric("Total Interactions", total_interactions)
                
                with col2:
                    success_rate = (df['error_count'] == 0).mean() * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                with col3:
                    avg_time = df['fix_time'].mean()
                    st.metric("Avg Execution Time", f"{avg_time:.2f}s")
                
                with col4:
                    total_errors = df['error_count'].sum()
                    st.metric("Total Errors", total_errors)

                st.subheader("📋 Recent Logs")
                # Show most recent logs with important columns
                display_columns = ['timestamp', 'prompt', 'error_count', 'fix_time', 'success']
                available_columns = [col for col in display_columns if col in df.columns]
                
                if available_columns:
                    st.dataframe(df[available_columns].head(10))
                else:
                    st.dataframe(df.head(10))

                # Time series analysis
                st.subheader("📊 Activity Over Time")
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    daily_activity = df.groupby(df['timestamp'].dt.date).size()
                    st.line_chart(daily_activity)

                # Error analysis
                st.subheader("🔍 Error Analysis")
                error_df = df[df['error_count'] > 0]
                if not error_df.empty:
                    st.write(f"**Error Patterns:** {len(error_df)} queries had errors")
                    
                    # Show error-prone queries
                    if 'prompt' in error_df.columns:
                        st.write("**Recent queries with errors:**")
                        for _, row in error_df.head(5).iterrows():
                            st.write(f"- {row['prompt'][:80]}... (Errors: {row['error_count']})")
                else:
                    st.success("🎉 No errors in the recorded logs!")
                        
            else:
                st.info("No entropy data recorded yet. Start interacting with the assistant to create logs.")

        except Exception as e:
            st.error(f"Could not read entropy logs: {e}")
            st.info("Trying alternative method...")
            
            # Fallback to direct SQLite access
            try:
                conn = sqlite3.connect("entropy_log.db")
                df_raw = pd.read_sql_query("SELECT * FROM logs", conn)
                conn.close()
                
                if not df_raw.empty:
                    st.subheader("Raw Database Logs")
                    st.dataframe(df_raw)
                else:
                    st.info("No data in entropy database.")
                    
            except Exception as sql_error:
                st.error(f"Could not access database directly: {sql_error}")
                
    else:
        st.info("Entropy log database not found. Start interacting with the assistant to create logs.")
        st.info("The database will be created automatically when you run your first analysis.")