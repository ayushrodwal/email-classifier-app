import streamlit as st
from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline(
    'sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")

# Streamlit Application
st.title("Email Classification and Analysis App")

# Input from the user
email_content = st.text_area("Enter the email content:")

# Function to classify email content
def classify_email(content):
    if "job opening" in content.lower() or "interview" in content.lower():
        return "Job-Related"
    elif "personal" in content.lower():
        return "Personal"
    elif "document" in content.lower() or "edit" in content.lower():
        return "Document Checking and Editing"
    else:
        return "Professional"

# Function to generate a reply based on classification
def generate_reply(content, classification):
    if classification == "Job-Related":
        return "Thank you for sharing this opportunity. I am interested in learning more about the job opening."
    elif classification == "Personal":
        return "Thank you for reaching out! I'll get back to you shortly."
    elif classification == "Document Checking and Editing":
        return "I've received your document. I'll review it and provide feedback soon."
    else:
        return "Thank you for the information. Looking forward to further communication."

# Analyze sentiment of the email content
def analyze_sentiment(content):
    sentiment_result = sentiment_pipeline(content)[0]
    label = sentiment_result['label']
    confidence = sentiment_result['score']
    return f"Sentiment: {label} (Confidence: {confidence:.2f})"

# Display results
if email_content:
    classification = classify_email(email_content)
    automated_reply = generate_reply(email_content, classification)
    sentiment = analyze_sentiment(email_content)

    st.subheader("Classification:")
    st.write(classification)

    st.subheader("Automated Reply:")
    st.write(automated_reply)

    st.subheader("Sentiment Analysis:")
    st.write(sentiment)
