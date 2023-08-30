# from wordcloud import WordCloud
pip install streamlit

import streamlit as st
from preprocess import load_data, perform_sentiment_analysis, load__cat_data
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import requests
import concurrent.futures

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False) # Hide Matplotlib deprecation warning

    st.title("Company Sentiment Analysis")
    # Load the datasets
    file_path = "C:/Users/Real_/OneDrive/Desktop/500_lv/Project/official/Yelp Senttiment_Analysis/archive(1)/yelp.csv"
    file_path_cat = "C:/Users/Real_/OneDrive/Desktop/500_lv/Project/official/Yelp Senttiment_Analysis/archive(1)/yelp_cat.csv"
    df = load_data(file_path)
    df_cat = load__cat_data(file_path_cat)

    min_date = df['date'].min()
    max_date = df['date'].max()


    with st.sidebar:
        # Dropdown input for selecting a company
        company_names = df['name'].unique()
        st.subheader("Input Options")
        selected_company = st.selectbox("Select a company:", company_names)

        # Date input for selecting start date
        start_date = st.date_input("Select start date", min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())

        # Date input for selecting end date
        end_date = st.date_input("Select end date", min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())

    # Display information about the selected company and its category
    selected_company_data = df_cat[df_cat['name'] == selected_company].iloc[0]
    selected_company_name = selected_company_data['name']
    selected_company_category = selected_company_data['categories']
    st.write(f"**Company: {selected_company_name}**")
    st.write(f"**Category: {selected_company_category}**")


    # Fetch and display Google description about the company
    google_api_key = "AIzaSyBINL2GFfTjbb3-eQFN_MnAE1R0MzNESzY"  # Replace with your Google API key
    google_search_url = "https://kgsearch.googleapis.com/v1/entities:search"
    params = {
        "query": selected_company_name,
        "limit": 1,
        "key": google_api_key,
    }
    google_response = requests.get(google_search_url, params=params)
    google_data = google_response.json()

    if "itemListElement" in google_data and len(google_data["itemListElement"]) >= 2:
        company_description1 = google_data["itemListElement"][0]["result"]["description"]
        company_description2 = google_data["itemListElement"][1]["result"]["description"]
        # st.subheader("Company Description 1 (Google)")
        st.write(f"**Company Description: {company_description1}**")
        # st.subheader("Company Description 2 (Google)")
        st.write(f"**Company Description: {company_description2}**")
    elif "itemListElement" in google_data and len(google_data["itemListElement"]) == 1:
        company_description = google_data["itemListElement"][0]["result"]["description"]
        # st.subheader("Company Description (Google)")
        st.write(f"**Company Description: {company_description}**")
    else:
        st.warning("No company description found.")


    # Display information about the date range
    st.write(f"**Sentiment analysis covers the date range: {min_date.date()} to {max_date.date()}**")


    # Convert start_date and end_date to datetime.datetime
    start_date = datetime.datetime.combine(start_date, datetime.time())
    end_date = datetime.datetime.combine(end_date, datetime.time())


    # Button to trigger sentiment analysis
    
    if st.button("Analyze Sentiment"):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(perform_sentiment_analysis, selected_company, start_date, end_date, df)
            sentiment_data = future.result()

        num_positive_reviews, num_neutral_reviews, num_negative_reviews, positive_reviews, neutral_reviews, negative_reviews = perform_sentiment_analysis(selected_company, start_date, end_date, df)
        
        # Filter the DataFrame for the selected company and within the date range
        company_filtered_df = df[(df['name'] == selected_company) & 
                                (df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Calculate the distribution of star ratings
        star_ratings_distribution = company_filtered_df['stars'].value_counts().sort_index()

        # Create a layout with two columns
        col1, col2 = st.columns(2)

        with col1:
            # Display the title for star ratings distribution
            st.subheader("Star Ratings Distribution")

            # Display the star ratings distribution
            st.bar_chart(star_ratings_distribution)
        
        
        if num_positive_reviews is not None:
            # Display the title for sentiment distribution
            st.subheader("Sentiment Distribution")
            sentiment_data = {
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Count': [num_positive_reviews, num_neutral_reviews, num_negative_reviews]
            }
            sentiment_df = pd.DataFrame(sentiment_data)
            st.bar_chart(sentiment_df.set_index('Sentiment'))

            # Create a DataFrame for the selected reviews
            reviews_table = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Review': [
                    positive_reviews[0] if positive_reviews else "No positive reviews",
                    neutral_reviews[0] if neutral_reviews else "No neutral reviews",
                    negative_reviews[0] if negative_reviews else "No negative reviews"
                ]
            })
            with col2:
                # Display the title for sentiment distribution
                st.subheader("Reviews based on sentiments")
                # st.write("Selected Reviews")
                st.dataframe(reviews_table)

        # st.subheader("Word Clouds")
    
        col3, col4 = st.columns(2)
    
        with col3:
            st.subheader("Word Cloud - Positive Reviews")
            if positive_reviews:
                positive_reviews_text = ' '.join(positive_reviews)
                positive_wordcloud = WordCloud(width=400, height=300, background_color="white").generate(positive_reviews_text)
                # st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.figure(figsize=(8, 6))
                plt.imshow(positive_wordcloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot()
            else:
                st.write("wow! No positive reviews.")

        with col4:
            st.subheader("Word Cloud - Negative Reviews")
            if negative_reviews:
                negative_reviews_text = ' '.join(negative_reviews)
                negative_wordcloud = WordCloud(width=400, height=300, background_color="white").generate(negative_reviews_text)
                # st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.figure(figsize=(8, 6))
                plt.imshow(negative_wordcloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot()
            else:
                st.write("Awesome! No negative reviews.")
        


if __name__ == "__main__":
    main()
