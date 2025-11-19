import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from transformers import pipeline
from streamlit_extras.metric_cards import style_metric_cards

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Security Log Analyser",
    page_icon="🛡️",
    layout="wide"
)

# Apply some global CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stSidebar {
        background-color: #161a23;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.image("https://img.icons8.com/external-flat-juicy-fish/344/external-cybersecurity-cyber-security-flat-flat-juicy-fish.png", width=60)
st.sidebar.title("🛡️ Security Log Analyser")
page = st.sidebar.radio("Navigation", ["🏠 Home", "📂 Upload Logs", "📊 Analysis"])

# -----------------------------
# Home Page
# -----------------------------
if page == "🏠 Home":
    st.title("🛡️ AI Security Log Analyser")
    st.markdown(
        """
        Welcome to the **AI-powered Security Log Analysis Dashboard** 🚀  

        ### Features:
        - 📂 Upload & preview raw security logs  
        - 📊 Visualize error/warning distribution  
        - 🚨 AI-based anomaly detection  
        - 🤖 Automated log summarization with NLP  
        """
    )
    st.success("Tip: Upload a `.log`, `.txt`, or `.csv` file to get started!")

# -----------------------------
# Upload Logs Page
# -----------------------------
elif page == "📂 Upload Logs":
    st.header("📂 Upload Security Log Files")
    uploaded_file = st.file_uploader("Upload a log file", type=["log", "txt", "csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except:
            lines = uploaded_file.read().decode("utf-8").splitlines()
            df = pd.DataFrame({"Logs": lines})

        st.subheader("Preview of Logs")
        st.dataframe(df.head(20), use_container_width=True)
        st.session_state["logs"] = df

# -----------------------------
# Analysis Page
# -----------------------------
elif page == "📊 Analysis":
    st.header("📊 Security Log Analysis Dashboard")

    if "logs" not in st.session_state:
        st.warning("⚠️ Please upload logs first!")
    else:
        df = st.session_state["logs"]

        # ================= Metrics =================
        total_logs = len(df)
        error_count = df[df.astype(str).apply(lambda x: x.str.contains("error", case=False).any(), axis=1)].shape[0]
        warning_count = df[df.astype(str).apply(lambda x: x.str.contains("warn", case=False).any(), axis=1)].shape[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("📑 Total Logs", total_logs)
        col2.metric("❌ Errors", error_count)
        col3.metric("⚠️ Warnings", warning_count)
        style_metric_cards(background_color="#262730", border_left_color="#FF4B4B")

        # ================= Chart =================
        st.subheader("📊 Error/Warning Distribution")
        counts = {"Errors": error_count, "Warnings": warning_count, "Normal": total_logs - (error_count + warning_count)}
        fig = px.pie(values=counts.values(), names=counts.keys(), hole=0.4,
                     color=counts.keys(), color_discrete_map={"Errors": "red", "Warnings": "orange", "Normal": "green"})
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

        # ================= Anomaly Detection =================
        with st.expander("🚨 Anomaly Detection", expanded=True):
            try:
                if "Logs" in df.columns:
                    vectorized = df["Logs"].astype(str).str.len().values.reshape(-1, 1)
                    clf = IsolationForest(contamination=0.1, random_state=42)
                    preds = clf.fit_predict(vectorized)
                    anomalies = df[preds == -1]

                    if anomalies.empty:
                        st.success("✅ No anomalies detected in logs.")
                    else:
                        st.error(f"⚠️ Found {len(anomalies)} anomalous log entries:")
                        st.dataframe(anomalies.head(10), use_container_width=True)
                else:
                    st.info("Logs format not supported for anomaly detection yet.")
            except Exception as e:
                st.warning(f"Anomaly detection failed: {e}")

        # ================= AI Insights =================
        with st.expander("🤖 AI Insights", expanded=True):
            try:
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                logs_text = " ".join(df.astype(str).head(50).values.flatten())  # first 50 lines
                summary = summarizer(logs_text, max_length=120, min_length=30, do_sample=False)
                st.info(summary[0]['summary_text'])
            except Exception as e:
                st.warning(f"AI summarization failed: {e}")
