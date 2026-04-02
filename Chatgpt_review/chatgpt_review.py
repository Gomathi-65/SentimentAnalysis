import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit as st
from wordcloud import WordCloud
import base64

# SAFE BACKGROUND IMAGE LOADER

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

background_img = r"c:\Users\Admin\Pictures\1675014214058.jpg"
img_base64 = get_base64_image(background_img)

# LOAD MODEL + VECTORIZER + DATA

MODEL_PATH = r"tuned_logistic_model.pkl"
VEC_PATH   = r"vectorizer_balanced.joblib"
DATA_PATH  = r"processed_cleaned_reviews.csv"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)
df_echo = pd.read_csv(DATA_PATH)

# PAGE CONFIG
st.set_page_config(page_title="AI Echo: Sentiment Analysis", layout="wide")

if img_base64:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}

        textarea {{
            background: black !important;
            color: white !important;
            border-radius: 10px !important;
        }}

        h1, h2, h3, h4, label {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# SIMPLE TEXT CLEANING

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# RULE-BASED WORDS

positive_words = {"good","great","excellent","awesome","love","fast","amazing","perfect","best"}
negative_words = {"bad","worst","slow","terrible","awful","problem","bug","crash","fail","error"}
neutral_words  = {"ok","okay","average","fine","normal","decent"}

# SMART HYBRID PREDICT FUNCTION

def smart_predict(text):
    cleaned = clean_text(text)
    words = cleaned.split()

    pos_hits = sum(1 for w in words if w in positive_words)
    neg_hits = sum(1 for w in words if w in negative_words)
    neu_hits = sum(1 for w in words if w in neutral_words)

    if pos_hits > neg_hits:
        return "positive"
    elif neg_hits > pos_hits:
        return "negative"
    elif neu_hits > 0:
        return "neutral"

    if len(words) <= 2:
        return "neutral"
    st.write(cleaned)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

# UI

st.title("📊 AI Echo — Sentiment Analysis")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🔮 Predict Sentiment", "📊 EDA"])

#  🔮 PREDICTION PAGE (EMOJI + BLACK RESULT)

if page == "🔮 Predict Sentiment":

    st.subheader("Enter a review:")
    user_text = st.text_area("Your review here...", key="review_input")

    if st.button("Predict"):

        if user_text.strip():

            pred = smart_predict(user_text)

            emoji_map = {
                "positive": "😊",
                "negative": "😡",
                "neutral": "😐"
            }

            emoji = emoji_map.get(pred.lower(), "🔍")

            st.markdown(
                f"""
                <div style="
                    background-color: white;
                    color: black;
                    padding: 18px;
                    border-radius: 12px;
                    font-size: 24px;
                    font-weight: bold;
                    text-align: center;
                    box-shadow: 0px 0px 12px black;
                    margin-top: 15px;
                ">
                    {emoji} Predicted Sentiment: {pred.upper()}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.warning("Please enter a review first.")

# 📊 EDA PAGE
elif page == "📊 EDA":
    st.sidebar.subheader("Exploratory Data Analysis (EDA)")
    st.sidebar.write("Explore the dataset and visualize sentiment distribution.")
    st.sidebar.markdown("---")

    st.title("📈 Exploratory Data Analysis (EDA)")
    st.write("This section allows you to explore the dataset and visualize sentiment distribution.")


    # dataloader
    if st.button("Load Data"):
        st.write("Data loaded successfully!")
        st.dataframe(df_echo.head(10))

    # Display dataset statistics
    st.subheader("Dataset Statistics")
    st.write("This dataset contains user reviews with their corresponding sentiments and ratings.")
    st.write(f"Total Reviews: {df_echo.shape[0]}")
    st.write(f"Total Columns: {df_echo.shape[1]}")
    st.write(f"Columns: {', '.join(df_echo.columns)}")
        # eda query
    query = st.selectbox("Select a query to visualize:", [
            "What is the overall sentiment of user reviews?",
            "How does sentiment vary by rating?",
            "Which keywords or phrases are most associated with each sentiment class?",
            "What is the distribution of sentiment across different ratings?",
            "How has sentiment changed over time?",
            "Do verified users tend to leave more positive or negative reviews?",
            "Are longer reviews more likely to be negative or positive?",
            "Which locations show the most positive or negative sentiment?",
            "Is there a difference in sentiment across platforms (Web vs Mobile)?",
            "Which ChatGPT versions are associated with higher/lower sentiment?",
            
        ])
    if st.button ("Run Query"):
        if query == "What is the overall sentiment of user reviews?":
            sentiment_counts = df_echo['sentiment'].value_counts()
            st.bar_chart(sentiment_counts, use_container_width=True)
            
        elif query == "How does sentiment vary by rating?":
            rating_sentiment = df_echo.groupby('rating')['sentiment'].value_counts().unstack().fillna(0)
            st.bar_chart(rating_sentiment)

        elif query == "Which keywords or phrases are most associated with each sentiment class?":
            sentiments = df_echo['sentiment'].unique()
            for sentiment in sentiments:
                reviews = df_echo[df_echo['sentiment'] == sentiment]['review'].dropna()
                if not reviews.empty:
                    text = " ".join(reviews)
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.subheader(f"Word Cloud for {sentiment} Sentiment")
                    st.pyplot(fig)
                else:
                    st.warning(f"No reviews found for {sentiment} sentiment.")

        elif query == "What is the distribution of sentiment across different ratings?":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=df_echo, x='rating', hue='sentiment', ax=ax)
            st.pyplot(fig)

        elif query == "How has sentiment changed over time?":
            if 'date' in df_echo.columns:
                dft = df_echo.dropna(subset=['date']).copy()
                dft['date'] = pd.to_datetime(dft['date'], errors='coerce')
                dft = dft.dropna(subset=['date'])
                dft["month"] = dft['date'].dt.to_period('M').dt.to_timestamp()
                sentiment_over_time = dft.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
                fig = px.line(
                    sentiment_over_time,
                    x=sentiment_over_time.index,
                    y=sentiment_over_time.columns,
                    labels={'value': 'Count', 'month': 'Month'},
                    title='Sentiment Over Time'
                )
                fig.update_layout(xaxis_title='Month', yaxis_title='Review Count')
                st.plotly_chart(fig)
            else:
                st.warning("The dataset does not contain a 'date' column for time-based analysis.")

        elif query == "Do verified users tend to leave more positive or negative reviews?":
            if 'verified_purchase' in df_echo.columns:
                verified_sentiment = df_echo.groupby('verified_purchase')['sentiment'].value_counts().unstack().fillna(0)
                verified_sentiment = verified_sentiment.reindex(['Yes', 'No'], axis=0, fill_value=0)
                st.bar_chart(verified_sentiment)
            else:
                st.warning("The dataset does not contain a 'verified_purchase' column for user verification status.")

        elif query == "Are longer reviews more likely to be negative or positive?":
            df_echo['review_length'] = df_echo['review'].apply(lambda x: len(str(x).split()))
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_echo, x='sentiment', y='review_length', ax=ax)
            st.pyplot(fig)

        elif query == "Which locations show the most positive or negative sentiment?":
            if 'location' in df_echo.columns:
                location_sentiment = df_echo.groupby('location')['sentiment'].value_counts().unstack().fillna(0)
                st.bar_chart(location_sentiment)
            else:
                st.warning("The dataset does not contain a 'location' column.")

        elif query == "Is there a difference in sentiment across platforms (Web vs Mobile)?":
            if 'platform_grouped' in df_echo.columns:
                platform_sentiment = df_echo.groupby('platform_grouped')['sentiment'].value_counts().unstack().fillna(0)
                platform_sentiment = platform_sentiment.reindex(['Web', 'Mobile'], axis=0, fill_value=0)
                st.bar_chart(platform_sentiment)
            else:
                st.warning("The dataset does not contain a 'platform_grouped' column.")               

        elif query == "Which ChatGPT versions are associated with higher/lower sentiment?":
            if 'version' in df_echo.columns:
                version_sentiment = df_echo.groupby('version')['sentiment'].value_counts().unstack().fillna(0)
                st.bar_chart(version_sentiment)
            else:
                st.warning("The dataset does not contain a 'version' column.")

        