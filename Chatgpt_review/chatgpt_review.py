import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

# PAGE CONFIG

st.set_page_config(
    page_title="Sentiment Analysis App",
    layout="wide"
)

# SIMPLE CLEAN STYLING

st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
h1, h2, h3 {
    color: #1f4e79;
}
[data-testid="metric-container"] {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# TITLE

st.title("📊 Sentiment Analysis & EDA Dashboard")

# LOAD MODEL

pipeline = pickle.load(open("sentiment_model.pkl", "rb"))

# LOAD DATA

df = pd.read_csv("cleaned_data.csv")
df.columns = df.columns.str.strip().str.lower()

# CREATE SENTIMENT COLUMN

def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df['sentiment'] = df['rating'].apply(get_sentiment)

# KPI METRICS SECTION

st.markdown("## 📌 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", len(df))
col2.metric("Average Rating ⭐", round(df['rating'].mean(), 2))
col3.metric("Positive % 😊", f"{round((df['sentiment']=='Positive').mean()*100,1)} %")
col4.metric("Negative % 😠", f"{round((df['sentiment']=='Negative').mean()*100,1)} %")

st.markdown("---")

# SENTIMENT PREDICTOR

st.header("🔍 Sentiment Predictor")

user_input = st.text_area("Enter a review:", height=120)

if st.button("Analyze Sentiment"):
    text = user_input.lower().strip().replace("\n", " ")

    # Keyword lists
    positive_words = ["good", "great", "excellent", "awesome", "nice", "amazing","well"]
    negative_words = ["bad", "worst", "poor", "terrible", "awful", "hate","not"]

    # Check if text contains any keyword
    if any(word in text.split() for word in positive_words):
        sentiment = "Positive"
        st.success(f"Predicted Sentiment: {sentiment}")

    elif any(word in text.split() for word in negative_words):
        sentiment = "Negative"
        st.error(f"Predicted Sentiment: {sentiment}")

    else:
        # ML model fallback
        prediction = pipeline.predict([text])[0]  # ensure 1-sample list
        label_map = {0: "Negative", 1: "Positive"}  # change if your model has Neutral
        sentiment = label_map[prediction]

        if sentiment == "Positive":
            st.success(f"Predicted Sentiment: {sentiment}")
        else:
            st.error(f"Predicted Sentiment: {sentiment}")

st.markdown("---")

# TEXT PREPROCESSING

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

df['clean_review'] = df['review'].apply(preprocess_text)
df['review_length'] = df['clean_review'].str.len()

# TABS

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overall Sentiment",
    "Sentiment vs Rating",
    "Keywords",
    "Verified Users",
    "Review Length",
    "Average Rating by Platform",
    "Average Rating by Major Version"
])

# TAB 1 - Overall Sentiment

with tab1:
    st.subheader("Overall Sentiment Distribution")
    fig = px.pie(
        df,
        names='sentiment',
        color='sentiment',
        color_discrete_map={'Positive':'green','Neutral':'gray','Negative':'red'}
    )
    st.plotly_chart(fig, use_container_width=True)

# TAB 2 - Sentiment vs Rating

with tab2:
    st.subheader("Sentiment vs Star Rating")
    rating_sentiment = pd.crosstab(df['rating'], df['sentiment'])
    fig = px.bar(rating_sentiment, barmode='stack')
    st.plotly_chart(fig, use_container_width=True)

# TAB 3 - WordCloud

with tab3:
    st.subheader("Word Cloud by Sentiment")
    sentiment_choice = st.selectbox("Choose Sentiment", ["Positive", "Neutral", "Negative"])
    text = " ".join(df[df['sentiment'] == sentiment_choice]['clean_review'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)

# TAB 4 - Verified Users

with tab4:
    st.subheader("Verified Purchase vs Sentiment")
    verified_sentiment = pd.crosstab(df['verified_purchase'], df['sentiment'])
    fig = px.bar(verified_sentiment, barmode='group')
    st.plotly_chart(fig, use_container_width=True)

# TAB 5 - Review Length

with tab5:
    st.subheader("Review Length vs Sentiment")
    avg_length = df.groupby('sentiment')['review_length'].mean().reset_index()
    fig = px.bar(avg_length, x='sentiment', y='review_length', color='sentiment')
    st.plotly_chart(fig, use_container_width=True)

# TAB 6 - Platform Analysis

with tab6:
    st.subheader("Average Rating by Platform")
    platform_avg_rating = df.groupby('platform')['rating'].mean().reset_index()
    fig = px.bar(platform_avg_rating, x='platform', y='rating', color='platform')
    st.plotly_chart(fig, use_container_width=True)

# TAB 7 - Version Analysis

with tab7:
    st.subheader("Average Rating by ChatGPT Major Version")
    df['major_version'] = df['version'].astype(str).str.split('.').str[0]
    avg_rating_major = df.groupby('major_version')['rating'].mean().reset_index()
    fig = px.bar(avg_rating_major, x='major_version', y='rating', color='major_version')
    st.plotly_chart(fig, use_container_width=True)
