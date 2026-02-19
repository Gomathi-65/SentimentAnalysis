📌 AI Echo: Your Smartest Conversational Partner

AI Echo is a sentiment analysis dashboard designed to analyze user reviews of ChatGPT-style applications. It classifies reviews as Positive, Neutral, or Negative, helping businesses understand customer satisfaction, monitor brand reputation, and optimize features.

🔹 Features

Sentiment Classification: Classifies reviews into Positive, Neutral, or Negative using NLP and Machine Learning.

Interactive Dashboard: Built with Streamlit for visualizing sentiment trends.

KPI Overview: Displays total reviews, average rating, positive %, and negative %.

EDA Insights: Explore sentiment distribution, review lengths, platform comparison, and version-based analysis.

Word Cloud Visualization: Shows frequently mentioned keywords for each sentiment.

Verified User Analysis: Compare sentiments between verified and non-verified users.

Platform & Version Insights: Understand user experience by platform (Web/Mobile) and ChatGPT version.

📊 Demo Screenshots

Dashboard Overview with KPI metrics

<img width="1912" height="944" alt="image" src="https://github.com/user-attachments/assets/2a57a345-0137-4ea9-ae1f-5f00b67c434a" />

🛠 Technologies & Libraries

Programming Language: Python

Libraries: pandas, numpy, nltk, scikit-learn, matplotlib, wordcloud, plotly, streamlit

Machine Learning Models: LinearSVC, TF-IDF, or any other text classification models

Deployment: Streamlit (optionally on AWS, Heroku, or Streamlit Cloud)

⚡ How It Works

Input a review in the text area.

Keyword check: Simple positive/negative word match.

ML fallback: If no keyword match, the trained ML model predicts sentiment.

EDA Tabs: Explore trends, ratings, word clouds, and platform/version insights.

📈 EDA Insights

Overall Sentiment Distribution: Pie chart of Positive, Neutral, Negative reviews.

Sentiment vs Rating: Stacked bar chart showing sentiment per rating.

Keywords: WordCloud of frequent words per sentiment.

Verified Users: Bar chart comparing verified vs non-verified users.

Review Length: Average review length per sentiment.

Platform & Version Analysis: Average rating by platform and major ChatGPT version.

📝 Project Deliverables

Cleaned & preprocessed dataset

Trained ML model (sentiment_model.pkl)

Streamlit dashboard (chatgpt_review.py)

Visualizations & WordCloud

Actionable insights report

📚 Skills Learned

Data Preprocessing & NLP Techniques

Exploratory Data Analysis (EDA)

Machine Learning & Deep Learning Models

Model Evaluation Metrics (Accuracy, Precision, Recall, F1-score)

Streamlit Deployment & Visualization

📌 Future Improvements

Add Neutral detection more robustly using multi-class models

Include Sentiment over time line chart

Deploy dashboard online with AWS / Streamlit Cloud

Include Topic Modeling for negative feedback themes

👨‍💻 Author

Created By: Gomathi Murugan
















