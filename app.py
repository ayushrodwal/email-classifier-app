import streamlit as st
import transformer
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")

import language_tool_python


# Load sentiment analysis pipeline
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")

sentiment_pipeline = load_sentiment_pipeline()

# Initialize grammar checker
@st.cache_resource
def load_language_tool():
    return language_tool_python.LanguageTool('en-US')

tool = load_language_tool()

# Email classification function
def classify_email(content):
    if "job opening" in content.lower() or "interview" in content.lower():
        return "Job-Related"
    elif "personal" in content.lower():
        return "Personal"
    elif "document" in content.lower() or "edit" in content.lower():
        return "Document Checking and Editing"
    else:
        return "Professional"

# Automated reply generator
def generate_reply(classification):
    if classification == "Job-Related":
        return "Thank you for sharing this opportunity. I am very interested in learning more about the job opening at Microsoft."
    elif classification == "Personal":
        return "Thank you for reaching out! I'll get back to you shortly."
    elif classification == "Document Checking and Editing":
        return "I've received your document. I'll review it and get back to you with feedback soon."
    else:
        return "Thank you for the information. I look forward to further communication."

# Sentiment analysis function
def analyze_sentiment(content):
    sentiment_result = sentiment_pipeline(content)[0]
    return f"{sentiment_result['label']} (Confidence: {sentiment_result['score']:.2f})"

# Grammar check function
def check_grammar(content):
    matches = tool.check(content)
    if not matches:
        return "No grammatical errors found."
    corrected_content = language_tool_python.utils.correct(content, matches)
    return f"Grammar issues found: {len(matches)}\nCorrected content:\n{corrected_content}"

# Streamlit App UI
st.title("Email Classifier and Analyzer")

email_content = st.text_area("Enter Email Content:", height=200)

if st.button("Analyze"):
    if email_content.strip():
        classification = classify_email(email_content)
        reply = generate_reply(classification)
        sentiment = analyze_sentiment(email_content)
        grammar = check_grammar(email_content)

        st.subheader("Results")
        st.write(f"**Classification:** {classification}")
        st.write(f"**Automated Reply:** {reply}")
        st.write(f"**Sentiment Analysis:** {sentiment}")
        st.write(f"**Grammar Check:** {grammar}")
    else:
        st.warning("Please enter email content to analyze!")
