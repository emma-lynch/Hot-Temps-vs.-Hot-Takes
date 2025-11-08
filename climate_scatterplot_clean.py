# climate_scatterplot_dynamic.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_scatter(show_plot=False, max_total_points=2000):
    """
    Scatterplot of sentiment vs average temperature with dynamic sampling.
    
    Parameters:
        show_plot (bool): If True, displays the plot interactively.
        max_total_points (int): Approximate maximum total points across all stances.
    """
    plt.close('all')  # close any existing figures

    # Load dataset
    df = pd.read_csv('climate_data.csv')
    df = df.dropna(subset=['sentiment', 'stance', 'temperature_avg'])
    df['stance'] = df['stance'].str.capitalize()

    # Determine dynamic sample size per stance
    num_stances = len(df['stance'].unique())
    sample_size_per_stance = math.ceil(max_total_points / num_stances)

    sampled_dfs = []
    for stance in df['stance'].unique():
        subset = df[df['stance'] == stance]
        sampled_subset = subset.sample(
            n=min(sample_size_per_stance, len(subset)),
            random_state=42
        )
        sampled_dfs.append(sampled_subset)

    df_sample = pd.concat(sampled_dfs)

    # Scatterplot with regression lines
    plt.figure(figsize=(12,7))
    colors = {'Believer':'blue', 'Non-believer':'red', 'Neutral':'green'}

    for stance, color in colors.items():
        if stance in df_sample['stance'].unique():
            subset = df_sample[df_sample['stance'] == stance]
            sns.regplot(
                data=subset,
                x='temperature_avg',
                y='sentiment',
                scatter=True,
                scatter_kws={'s':20, 'alpha':0.5, 'color':color},
                line_kws={'color':color, 'label':f'{stance} Trend'},
                ci=None
            )

    # Labels, title, legend
    plt.xlabel('Average Temperature (°C)', fontsize=12)
    plt.ylabel('Tweet Sentiment Score', fontsize=12)
    plt.title('Temperature vs Climate Change Sentiment (Dynamic Sample)', fontsize=14)
    plt.legend(title='Stance')

    # Save and optionally show
    plt.tight_layout()
    plt.savefig('scatter_sentiment_temperature_dynamic.png', dpi=200)
    
    if show_plot:
        plt.show()

    plt.close()
    print("Scatterplot saved to 'scatter_sentiment_temperature_dynamic.png'.")

if __name__ == "__main__":
    plot_scatter(show_plot=True)
