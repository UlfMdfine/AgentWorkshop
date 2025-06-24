import json
import os
from pathlib import Path

import mammoth
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.agents import supervisor

# Streamlit configuration
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "2000"

# Directory setup
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_latest_message():
    """Retrieve the latest message from the chat history."""
    if st.session_state.chat_history:
        entry = st.session_state.chat_history[-1]
        if isinstance(entry, dict) and "Supervisor" in entry:
            messages = entry["Supervisor"].get("messages", [])
            if messages:
                return messages[-1].content
        elif isinstance(entry, dict) and "content" in entry:
            return entry["content"]
    return ""


st.set_page_config(page_title="Agentic Report Generator", layout="wide")


# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 Agentic Review of Estimates Report Generator")

# Tabs
tab1, tab2, tab3 = st.tabs(["Data Viewer", "Report Generation", "Thinking Process"])

# ---------------- Tab 1: Data Upload ----------------
with tab1:
    # uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    # if uploaded_file:
    file_path = "synthetic_credit_risk_data_20250606.csv"
    # with open(file_path, "wb") as f:
    #    f.write(uploaded_file.getbuffer())
    # st.success("File uploaded successfully!")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, delimiter=",")
        st.write(f"**Preview of {file_path}:**")
        st.dataframe(df.head())
    else:
        st.warning("Please upload a CSV file to get started.")

# ---------------- Tab 2: Report Creation ----------------
with tab2:
    if os.path.exists(file_path):
        csv_path = file_path
        user_instruction = st.text_area("Add any context/instruction for the report:")

        clear_history = st.checkbox(
            "Start fresh and clear previous conversation history",
            value=True,
            help="Only uncheck this if you want to continue from the latest supervisor message.",
        )

        if st.button("Generate"):
            # Clear previous outputs
            for file in OUTPUT_DIR.glob("*"):
                try:
                    file.unlink()
                except Exception as e:
                    st.warning(f"Failed to delete {file.name}: {e}")

            message = {
                "role": "user",
                "content": f"Please use the CSV file at '{csv_path}' as input, if applicable. {user_instruction}",
            }

            if clear_history or "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            history = []
            latest_assistant_msg = get_latest_message()
            if latest_assistant_msg:
                history.append({"role": "assistant", "content": latest_assistant_msg})
            history.append({"role": "user", "content": message["content"]})

            with st.spinner("Generating report... Please wait."):
                for chunk in supervisor.stream({"messages": history}):
                    st.session_state.chat_history.append(chunk)

            report_docx = OUTPUT_DIR / "AI_Agent_Summary_Final.docx"
            if report_docx.exists():
                st.success("Report generated successfully.")
                with open(report_docx, "rb") as docx_file:
                    result = mammoth.convert_to_html(docx_file)
                    html = result.value  # The generated HTML

                components.html(html, height=600, scrolling=True)

                with open(report_docx, "rb") as f:
                    st.download_button(
                        "Export report (.docx)", f, file_name=report_docx.name
                    )
            else:
                st.warning(
                    "No report generated. Please check the shown supervisor message."
                )
                st.markdown(get_latest_message())
    else:
        st.warning("Please upload a CSV file in the Data Viewer tab to get started.")

# ---------------- Tab 3: Thinking Process ----------------
with tab3:

    def render_tool_calls(tool_calls):
        for call in tool_calls:
            st.markdown("**🔧  Tool Call Details:**")
            if call.get("name"):
                st.markdown(f"- *Function*: `{call.get('name')}`")
            if call.get("args"):
                st.markdown(
                    f"- *Arguments*:\n```json\n{json.dumps(call.get('args', {}), indent=2)}\n```"
                )

    def render_usage(meta):
        if meta and "token_usage" in meta:
            usage = meta["token_usage"]
            st.markdown(
                f"📊 **Tokens Used**: Prompt `{usage.get('prompt_tokens', 0)}`, Completion `{usage.get('completion_tokens', 0)}`, Total `{usage.get('total_tokens', 0)}`"
            )

    def render_step_header(idx, msg):
        step = f"Step {idx + 1}"
        if isinstance(msg, HumanMessage):
            return f"{step}:&nbsp;&nbsp;&nbsp;&nbsp;**Human Message 👩‍💻**"
        elif isinstance(msg, ToolMessage):
            return (
                f"{step}:&nbsp;&nbsp;&nbsp;&nbsp;**Tool Response from `{msg.name}` 🛠️**"
            )
        elif isinstance(msg, AIMessage):
            return (
                f"{step}:&nbsp;&nbsp;&nbsp;&nbsp;**AI Message from `{msg.name}` 🤖**"
                if msg.name
                else f"{step}:&nbsp;&nbsp;&nbsp;&nbsp;**AI Message 🤖**"
            )
        else:
            return "❓ Unknown message type"

    def render_message(msg):
        if isinstance(msg, HumanMessage):
            with st.container():
                st.markdown("**User Input:**")
                st.markdown(msg.content)

        elif isinstance(msg, ToolMessage):
            st.markdown("**Response**:")
            st.markdown(msg.content)

        elif isinstance(msg, AIMessage):
            if msg.content:
                st.markdown("**Content**:")
                st.markdown(msg.content)
            if msg.tool_calls:
                render_tool_calls(msg.tool_calls)
            if hasattr(msg, "response_metadata"):
                render_usage(msg.response_metadata)

        else:
            st.markdown("❓ Unknown message type")

    if st.session_state.chat_history:
        final_supervisor_entry = st.session_state.chat_history[-1]
        chat_history = final_supervisor_entry.get("Supervisor", []).get("messages", [])

        for i, message in enumerate(chat_history):
            with st.expander(render_step_header(i, message), expanded=False):
                render_message(message)
    else:
        st.warning(
            "No thinking process of agents to display. Please generate a report first."
        )
