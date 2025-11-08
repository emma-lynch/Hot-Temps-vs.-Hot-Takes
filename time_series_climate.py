import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv(r'c:\Users\16093\OneDrive\Documents\chromedriver-win64\Climate Project\climate_data.csv')

# Ensure 'created_at' is datetime
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df = df.dropna(subset=['created_at'])

# Clean 'topic' (strip whitespace and remove empty strings)
df['topic'] = df['topic'].astype(str).str.strip()
df = df[df['topic'] != '']

# Check that sentiment is numeric
df['sentiment_num'] = pd.to_numeric(df['sentiment'], errors='coerce')
df = df.dropna(subset=['sentiment_num'])

# Aggregate by week
df['week'] = df['created_at'].dt.to_period('W').dt.to_timestamp()

# Select top 5 topics by number of tweets
top_topics = df['topic'].value_counts().nlargest(5).index
df_top = df[df['topic'].isin(top_topics)]

# Compute average sentiment per week per topic
weekly_sentiment_top = df_top.groupby(['week', 'topic'])['sentiment_num'].mean().reset_index()

# Plot
plt.figure(figsize=(12,6))
sns.lineplot(
    data=weekly_sentiment_top, 
    x='week', 
    y='sentiment_num', 
    hue='topic', 
    marker='o'
)
plt.title('Average Sentiment Over Time for Top 5 Topics')
plt.xlabel('Week')
plt.ylabel('Average Sentiment')
plt.xticks(rotation=45)
plt.legend(title='Topic', loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout()
plt.show()
