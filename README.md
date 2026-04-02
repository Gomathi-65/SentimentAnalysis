# 📊 AI Echo — Sentiment Analysis App

## 📌 Project Overview
AI Echo is a hybrid sentiment analysis system that classifies user reviews into Positive, Negative, or Neutral using:

- Machine Learning (TF-IDF + Logistic Regression)  
- Rule-based NLP approach  

It also includes a Streamlit web app for real-time predictions and EDA visualization.

---

## 🎯 Objectives
- Analyze user reviews from multiple platforms  
- Build an accurate sentiment classifier  
- Improve accuracy using hybrid approach  
- Provide real-time predictions  

---

## 📂 Dataset
- Total records: 250  
- Features:
  - review, rating, platform, location  
  - version, helpful_votes  
  - verified_purchase  

---

## ⚙️ Tech Stack
- Python  
- pandas, numpy  
- nltk  
- scikit-learn  
- matplotlib, seaborn, plotly  
- wordcloud  
- streamlit  

---

## 🔄 Workflow

### 1. Data Preprocessing
- Lowercasing  
- Removing special characters  
- Stopword removal (keeping negations)  
- Lemmatization  

### 2. Feature Engineering
- Created cleaned_reviews  
- Converted ratings to sentiment:
  - 4–5 → Positive  
  - 3 → Neutral  
  - 1–2 → Negative  

### 3. Data Balancing
- Used upsampling to balance dataset  

### 4. Model Building
- TF-IDF Vectorizer  
- Logistic Regression  

### 5. Model Performance
- Before tuning: 76%  
- After tuning: 81%  

### 6. Hybrid Model
- Rule-based prediction first  
- ML model as fallback  

---

## 💻 Streamlit Features

### 🔮 Prediction Page
- Input review  
- Output sentiment with emoji  

### 📊 EDA Page
- Sentiment distribution  
- Rating vs sentiment  
- WordCloud  
- Location analysis  
- Version analysis  

---

## 📁 Project Structure
AI-Echo/
│
├── processed_cleaned_reviews.csv  
├── tuned_logistic_model.pkl  
├── vectorizer_balanced.joblib  
│
├── app.py  
├── notebook.ipynb  
│
└── README.md  

---

## 🚀 How to Run

git clone <your-repo-link>  
cd AI-Echo  

pip install -r requirements.txt  

streamlit run app.py  

---

## ⚠️ Challenges
- Invalid date values  
- Imbalanced dataset  
- Short text prediction issues  
- Solver error in Logistic Regression  

---

## 🔮 Future Improvements
- Use deep learning models  
- Add multilingual support  
- Deploy on cloud  

---

## 👩‍💻 Author
Gomathi Murugan  

---

## ⭐ Conclusion
This project combines rule-based NLP and machine learning to provide accurate sentiment predictions.
