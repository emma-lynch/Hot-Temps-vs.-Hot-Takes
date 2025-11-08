# climate_heatmap_volume.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_heatmap(csv_file='climate_data.csv', nrows=None, save_file='sentiment_heatmap_volume.png',
                     sample_random=False, pivot_csv='monthly_sentiment_counts.csv',
                     start_date='2006-01-01', end_date='2011-12-31'):
    """
    Generates a heatmap of sentiment distribution over time (by month) from a climate dataset.
    Adds tweet volume overlay line for context.
    """

    # Load precomputed pivot table if available
    if os.path.exists(pivot_csv) and not sample_random and nrows is None:
        print(f"Loading precomputed pivot table from '{pivot_csv}'...")
        heatmap_data = pd.read_csv(pivot_csv, index_col=0)
        heatmap_data.columns = pd.to_datetime(heatmap_data.columns)
    else:
        print("Processing dataset to compute monthly sentiment counts...")
        df = pd.read_csv(csv_file, nrows=None if sample_random else nrows)

        if sample_random and nrows is not None:
            df = df.sample(n=nrows, random_state=42)

        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df = df.dropna(subset=['created_at'])

        try:
            if df['created_at'].dt.tz is not None:
                df['created_at'] = df['created_at'].dt.tz_convert(None)
        except Exception:
            df['created_at'] = df['created_at'].dt.tz_localize(None, ambiguous='NaT', nonexistent='NaT')

        if df.empty:
            print("No valid dates found. Exiting.")
            return

        # Sentiment categorization
        def categorize_sentiment(value):
            if value < -0.05:
                return 'Negative'
            elif value > 0.05:
                return 'Positive'
            else:
                return 'Neutral'
        df['sentiment_label'] = df['sentiment'].apply(categorize_sentiment)

        # Filter by date range
        df = df[(df['created_at'] >= pd.to_datetime(start_date)) &
                (df['created_at'] <= pd.to_datetime(end_date))]

        df['month'] = df['created_at'].dt.to_period('M').apply(lambda r: r.start_time)
        sentiments = ['Negative', 'Neutral', 'Positive']
        all_months = pd.date_range(start=start_date, end=end_date, freq='MS')

        heatmap_data = df.pivot_table(
            index='sentiment_label',
            columns='month',
            aggfunc='size',
            fill_value=0
        ).reindex(index=sentiments, columns=all_months, fill_value=0)

        if not sample_random and nrows is None:
            heatmap_data.to_csv(pivot_csv)
            print(f"Pivot table saved to '{pivot_csv}'.")

    # 🔹 ADDED: compute tweet volume per month
    tweet_volume = heatmap_data.sum(axis=0)

    # Plot heatmap with overlay
    fig, ax1 = plt.subplots(figsize=(20, 6))
    sns.heatmap(
        heatmap_data,
        cmap='coolwarm',
        linewidths=0.5,
        annot=False,
        fmt='d',
        vmin=0,
        vmax=max(heatmap_data.values.max(), 1),
        ax=ax1,
        cbar_kws={'label': 'Tweet Count'}
    )

    # Adjust x-axis
    nth_month = max(1, heatmap_data.shape[1] // 20)
    x_labels = [str(month.date()) if i % nth_month == 0 else '' for i, month in enumerate(heatmap_data.columns)]
    ax1.set_xticks(range(len(heatmap_data.columns)))
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Sentiment')
    ax1.set_title("Climate Change Sentiment & Tweet Volume (2006–2011)", fontsize=14)

    # 🔹 ADD SECONDARY AXIS for tweet volume
    ax2 = ax1.twinx()
    ax2.plot(range(len(tweet_volume)), tweet_volume, color='black', linewidth=2, alpha=0.7, label='Tweet Volume')
    ax2.set_ylabel('Total Tweets per Month', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # 🔹 Optional legend
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_file, dpi=200)
    plt.show()

    print(f"Heatmap with tweet volume overlay saved to '{save_file}'.")


# Example usage
if __name__ == "__main__":
    generate_heatmap(csv_file='climate_data.csv',
                     save_file='sentiment_heatmap_volume_2006_2011.png',
                     start_date='2006-01-01', end_date='2011-12-31')
