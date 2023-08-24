import streamlit as st
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import re
import datetime
# import plotly.express as px
import random

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load__cat_data(file_path):
    df = pd.read_csv(file_path)
    return df


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, URLs, and mentions
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    # Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Join tokens back to form the cleaned text
    return ' '.join(tokens)
        

@st.cache_data
def perform_sentiment_analysis(company_name, start_date, end_date, df):
    # Lists to store sentiment scores
    positive_tweets = []
    neutral_tweets = []
    negative_tweets = []


    # Filter the DataFrame to include only rows with the company name  and within the date range
    company_filtered_df = df[(df['name'] == company_name) &
                             (df['date'] >= start_date) & (df['date'] <= end_date)]

    # If there are no tweets within the date range, display a message to the user
    if len(company_name) == 0:
        st.warning(f"No reviews found for '{company_name}' within the specified date range.")
        st.info("Please adjust the date range or enter a different company name.")
        return None, None, None, None


    company_filtered_df['Sentiment'] = company_filtered_df['text'].apply(
        lambda text: TextBlob(preprocess_text(text)).sentiment.polarity
    )

    positive_reviews = company_filtered_df[company_filtered_df['Sentiment'] > 0.2]['text'].tolist()
    neutral_reviews = company_filtered_df[(company_filtered_df['Sentiment'] >= -0.2) & (company_filtered_df['Sentiment'] <= 0.2)]['text'].tolist()
    negative_reviews = company_filtered_df[company_filtered_df['Sentiment'] < -0.2]['text'].tolist()

    num_positive_reviews = len(positive_reviews)
    num_neutral_reviews = len(neutral_reviews)
    num_negative_reviews = len(negative_reviews)

    return (
        num_positive_reviews, num_neutral_reviews, num_negative_reviews,
        positive_reviews, neutral_reviews, negative_reviews
    )

    
