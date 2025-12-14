import streamlit as st
from src.predict import predict_sentiment
import time

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTextArea {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Sentiment Analysis App (BERT)")
st.caption("Real-time sentiment detection powered by BERT")
st.divider()

st.subheader("Enter Your Text")
user_text = st.text_area(
    "Text to analyze",
    placeholder="Type or paste your text here...",
    height=150,
    label_visibility="collapsed"
)

char_count = len(user_text)
st.caption(f"📝 {char_count} characters")

col1, col2 = st.columns([1, 3])

with col1:
    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

if analyze_btn:
    if user_text.strip():
        with st.spinner("🤔 Analyzing sentiment..."):
            time.sleep(0.5)
            result = predict_sentiment(user_text)
        st.divider()
        st.subheader("Result")
        result_lower = result.lower()
        
        if "positive" in result_lower:
            st.success(f"😊 **{result}**")
        elif "negative" in result_lower:
            st.error(f"😞 **{result}**")
        else:
            st.warning(f"😐 **{result}**")
        
    else:
        st.warning("⚠️ Please enter some text to analyze")