🧠 AI Echo – Sentiment Analysis of ChatGPT Reviews

AI Echo is a machine learning based Sentiment Analysis Web Application that analyzes user reviews of a ChatGPT-style application and classifies them into Positive, Neutral, or Negative sentiments. The project also provides rich visual insights using EDA dashboards.

🚀 Features

Classifies user reviews into Positive / Neutral / Negative

Text preprocessing using spaCy NLP pipeline

TF-IDF based feature extraction

ML model prediction (Logistic Regression / XGBoost)

Interactive Streamlit dashboard

Visual Insights:

Overall sentiment distribution

Sentiment by rating

Positive & Negative WordClouds

Sentiment trends over time

Platform & Location based sentiment

Verified vs Non-verified user analysis

Common negative feedback themes

📂 Project Structure AI-Echo-Sentiment-Analysis/ │ ├── chat_gpt.py # Streamlit application ├── AI.ipynb # Model training notebook ├── clean.csv # Cleaned dataset ├── tfidf_vectorizer.pkl # Saved TF-IDF Vectorizer ├── sentiment_model.pkl # Trained ML model ├── README.md # Project documentation └── requirements.txt # Project dependencies

⚙ Installation & Setup 1️⃣ Create Virtual Environment python -m venv venv venv\Scripts\activate

2️⃣ Install Dependencies pip install pandas numpy scikit-learn nltk spacy wordcloud matplotlib seaborn xgboost streamlit openpyxl langdetect imbalanced-learn python -m spacy download en_core_web_sm

▶ Run the Application streamlit run chat_gpt.py

The app will open automatically in your browser.

📊 Dataset Information

The dataset contains:

Column Description date Review submission date title Review headline review Full review text rating User rating (1–5) username Random username helpful_votes Number of helpful votes platform Web / Mobile language Language of review location Country version ChatGPT version verified_purchase Yes / No 📈 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

🧠 Model Techniques Used

NLP Preprocessing – Tokenization, Lemmatization, Stopword Removal

Feature Extraction – TF-IDF

Models – Logistic Regression, XGBoost

Visualization – WordCloud, Seaborn, Matplotlib

🎯 Business Use Cases

Customer feedback analysis

Brand reputation tracking

Feature improvement recommendations

Automated complaint detection

Product satisfaction monitoring

👩‍💻 Developed By

Gomathi Murugan Capstone Project – AI Echo: Sentiment Analysis of ChatGPT Reviews
