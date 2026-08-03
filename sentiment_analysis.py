import pandas as pd
import os
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import nltk

nltk.download('vader_lexicon')

# Load data
filename = "covid19_tweets.csv"
df = pd.read_csv(filename)
print(f"✅ Loaded {filename} with shape: {df.shape}")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Identify tweet column
tweet_col = next((col for col in df.columns if "tweet" in col), None)
if not tweet_col:
    raise ValueError("Tweet column not found.")

# Filter healthcare workers
hcw_keywords = ['doctor', 'nurse', 'surgeon', 'physician', 'healthcare', 'medic', 'paramedic']
df['user_description'] = df['user_description'].astype(str).str.lower()
mask = df['user_description'].apply(lambda x: any(keyword in x for keyword in hcw_keywords))
filtered_df = df[mask].copy()
print(f"📊 Total filtered HCW tweets: {filtered_df.shape[0]}")
filtered_df.to_csv('healthcare_worker_tweets.csv', index=False)
print("✅ Saved filtered data to 'healthcare_worker_tweets.csv'")

# Sentiment Analysis
sid = SentimentIntensityAnalyzer()
def get_sentiment(text):
    if pd.isnull(text):
        return 'Neutral'
    score = sid.polarity_scores(text)['compound']
    return 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'

filtered_df['Sentiment'] = filtered_df[tweet_col].apply(get_sentiment)
filtered_df.to_csv('sentiment_analysis_results.csv', index=False)
print("✅ Saved sentiment analysis results to 'sentiment_analysis_results.csv'")

# SENTIMENT OVER TIME
filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce', dayfirst=True)
filtered_df.dropna(subset=['date'], inplace=True)

sentiment_over_time = filtered_df.groupby(filtered_df['date'].dt.to_period('M'))['Sentiment'].value_counts().unstack().fillna(0)
sentiment_over_time.plot(kind='line', marker='o', figsize=(10, 6))
plt.title('Sentiment Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Tweets')
plt.grid(True)
plt.tight_layout()
plt.show()
input("Press Enter to continue...")

# COMPARISON: Doctors vs Nurses
def classify_role(desc):
    desc = str(desc).lower()
    if 'doctor' in desc or 'physician' in desc:
        return 'Doctor'
    elif 'nurse' in desc:
        return 'Nurse'
    else:
        return 'Other'

filtered_df['Role'] = filtered_df['user_description'].apply(classify_role)
role_sentiment = filtered_df[filtered_df['Role'].isin(['Doctor', 'Nurse'])].groupby(['Role', 'Sentiment']).size().unstack().fillna(0)

role_sentiment.plot(kind='bar', figsize=(8, 6))
plt.title('Sentiment Comparison: Doctors vs Nurses')
plt.ylabel('Number of Tweets')
plt.tight_layout()
plt.show()
input("Press Enter to continue...")

# WORD CLOUDS by Sentiment
for sentiment in ['Positive', 'Negative', 'Neutral']:
    text_data = ' '.join(filtered_df[filtered_df['Sentiment'] == sentiment][tweet_col].dropna())
    wc = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Tweets')
    plt.tight_layout()
    plt.show()
    input("Press Enter to continue...")

