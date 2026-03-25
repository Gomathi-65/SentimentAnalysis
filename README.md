<h1>DS_AI-Echo-Your-Smartest-Conversational-Partner</h1>
<h1>🤖 DS_AI-Echo — Your Smartest Conversational Partner</h1>
<h2>📊 Sentiment Analysis & Insights Dashboard (Streamlit + ML + NLP)</h2>
AI Echo is an interactive <b>Sentiment Analysis Dashboard</b> built using <b>Streamlit, Machine Learning,</b> and <b>NLP techniques.</b>
It analyzes user reviews, predicts sentiment (Positive, Neutral, Negative), and provides powerful EDA visualizations to understand user feedback deeply.

<h1>🚀 Features</h1>
<h3>🔍 1. Sentiment Classification</h3>
Uses ML model (Logistic Regression + TF-IDF)
Uses VADER/TextBlob hybrid for word-level + sentence-level sentiment
Supports:
<B>Positive</B>
<B>Neutral</B>
<B>Negative</B>
<Br>

<h1>🎨 2. Modern UI & Dashboard</h1>
Futuristic AI-themed background
Black glass review input box
Responsive, clean design
📈 3. Interactive Visualizations
Includes multiple insights:

Sentiment Insights
Overall sentiment distribution
Sentiment by rating
Sentiment by platform (Web / Mobile)
Sentiment by ChatGPT version
Sentiment by user location
Sentiment over time (monthly trends)
Verified vs non-verified sentiment
Text Insights
Word clouds for each sentiment
Common negative feedback themes
Review length distribution by sentiment
🧠 Machine Learning Pipeline
✔ Preprocessing:
Lowercasing
Special character removal
Stopword removal (negators kept)
Lemmatization using WordNet
POS-aware normalization
Missing value handling
Platform grouping (Web, Mobile, Other)
✔ Sentiment Labeling:
Based on Rating

>= 4 → Positive
3 → Neutral
<= 2 → Negative
Additional VADER-based compound scoring

Word-level hybrid sentiment rules

Final ensemble sentiment classification

✔ Model Training:
Balanced dataset (upsampling)
TF-IDF Vectorizer (1–2 grams, 20k features)
Logistic Regression classifier
Stratified train-test split (80/20)
Saved models using joblib:
sentiment_analyzer.joblib
vectorizer_balanced.joblib
text_classifier_balanced.joblib
📁 Dataset Requirements
Your dataset should contain at least:

Column	Description
review	User review text
rating	Rating 1–5
date	Review date
verified_purchase	Yes/No
location	User's location
platform	App Store, Play Store, Web, etc.
version	ChatGPT version
cleaned_reviews	Preprocessed cleaned text
sentiment	Final sentiment label
