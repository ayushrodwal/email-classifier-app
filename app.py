import streamlit as st

import language_tool_python
import os

from transformers import pipeline

# Cache models and resources to avoid reloading
@st.cache_resource
def load_sentiment_pipeline():
    # Use a lightweight model for sentiment analysis
    return pipeline('sentiment-analysis', model="distilbert-base-uncased")

@st.cache_resource
def load_language_tool():
    # Load the grammar checker
    return language_tool_python.LanguageTool('en-US')

sentiment_pipeline = load_sentiment_pipeline()
tool = load_language_tool()

# Email classification function
def classify_email(content):
    content_lower = content.lower()
    if "job opening" in content_lower or "interview" in content_lower:
        return "Job-Related"
    elif "personal" in content_lower:
        return "Personal"
    elif "document" in content_lower or "edit" in content_lower:
        return "Document Checking and Editing"
    else:
        return "Professional"

# Automated reply generator
def generate_reply(classification):
    replies = {
        "Job-Related": "Thank you for sharing this opportunity. I am very interested in learning more about the job opening.",
        "Personal": "Thank you for reaching out! I'll get back to you shortly.",
        "Document Checking and Editing": "I've received your document. I'll review it and get back to you with feedback soon.",
        "Professional": "Thank you for the information. I look forward to further communication."
    }
    return replies.get(classification, "Thank you for your email.")

# Sentiment analysis function
def analyze_sentiment(content):
    try:
        sentiment_result = sentiment_pipeline(content)[0]
        return f"{sentiment_result['label']} (Confidence: {sentiment_result['score']:.2f})"
    except Exception as e:
        return f"Error in sentiment analysis: {str(e)}"

# Grammar check function
def check_grammar(content):
    try:
        matches = tool.check(content)
        if not matches:
            return "No grammatical errors found."
        corrected_content = language_tool_python.utils.correct(content, matches)
        return f"Grammar issues found: {len(matches)}\nCorrected content:\n{corrected_content}"
    except Exception as e:
        return f"Error in grammar check: {str(e)}"

# Streamlit App UI
st.title("Email Classifier and Analyzer")

email_content = st.text_area("Enter Email Content:", height=200)

if st.button("Analyze"):
    if email_content.strip():
        # Perform analysis
        classification = classify_email(email_content)
        reply = generate_reply(classification)
        sentiment = analyze_sentiment(email_content)
        grammar = check_grammar(email_content)

        # Display results
        st.subheader("Results")
        st.write(f"**Classification:** {classification}")
        st.write(f"**Automated Reply:** {reply}")
        st.write(f"**Sentiment Analysis:** {sentiment}")
        st.write(f"**Grammar Check:** {grammar}")
    else:
        st.warning("Please enter email content to analyze!")
