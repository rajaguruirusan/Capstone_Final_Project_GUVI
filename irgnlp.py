import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

# Download necessary NLTK resources
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('punkt_tab')

# Define Streamlit layout
st.title("GUVI Final Project: NLP Analysis - IRG")

# Input section
st.subheader("Enter your text here:")
input_texts = st.text_area(label="Input Texts", value="\n", height=200)

# Add Submit button and handle input validation
if st.button("Analyze"):
    if not input_texts.strip():
        st.error("Please enter text to analyze.")
    else:
        # Tokenization
        st.subheader("Stemming")
        tokens = word_tokenize(input_texts)
        ps = PorterStemmer()
        stemmed_words = [ps.stem(word) for word in tokens if word.isalnum()]
        st.write(" ".join(stemmed_words))

        # Named Entity Recognition (NER)
        st.subheader("Named Entity Recognition (NER)")
        ner_text = TextBlob(input_texts)
        st.write(ner_text.noun_phrases)

        # Keyword Extraction
        st.subheader("Keyword Extraction")
        vectorizer = CountVectorizer(max_features=5, stop_words='english')
        keywords = vectorizer.fit_transform([input_texts])
        keyword_list = vectorizer.get_feature_names_out()
        st.write(keyword_list)

        # Sentiment Analysis
        st.subheader("Sentiment Analysis")
        sentiment = TextBlob(input_texts).sentiment
        st.write(f"Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")

        # Sentiment Visualization
        sentiment_labels = ["Positive", "Neutral", "Negative"]
        sentiment_counts = [
            len([1 for sentence in input_texts.split('.') if TextBlob(sentence).sentiment.polarity > 0]),
            len([1 for sentence in input_texts.split('.') if TextBlob(sentence).sentiment.polarity == 0]),
            len([1 for sentence in input_texts.split('.') if TextBlob(sentence).sentiment.polarity < 0]),
        ]

        plt.figure(figsize=(8, 4))
        plt.bar(sentiment_labels, sentiment_counts)
        plt.title("Sentiment Analysis Results")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        st.pyplot(plt)

        # Word Cloud
        st.subheader("Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(input_texts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
