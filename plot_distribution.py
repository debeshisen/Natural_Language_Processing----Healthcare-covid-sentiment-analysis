import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('sentiment_analysis_results.csv')

# Count the number of each sentiment
sentiment_counts = df['Sentiment'].value_counts()

# Create the bar graph
plt.figure(figsize=(10, 6))
bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
               color=['green', 'blue', 'red'], alpha=0.7)

# Add counts on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

# Customize the graph
plt.title('Sentiment Distribution of Tweets', fontsize=16)
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Number of Tweets', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()