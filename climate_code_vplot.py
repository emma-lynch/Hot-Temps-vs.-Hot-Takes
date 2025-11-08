# climate_violin_plot_single.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_violin(show_plot=False):
    plt.close('all')  # close any existing figures

    # Load dataset
    df = pd.read_csv('climate_data.csv')
    df = df.dropna(subset=['sentiment', 'stance'])
    df['stance'] = df['stance'].str.capitalize()

    # Create plot
    plt.figure(figsize=(8,6))
    ax = sns.violinplot(
        data=df,
        x='stance',
        y='sentiment',
        palette='coolwarm',
        inner='point',
        dodge=False
    )

    ax.set_xlabel('Stance', fontsize=12)
    ax.set_ylabel('Sentiment Score', fontsize=12)
    ax.set_title('Overall Distribution of Climate Change Sentiment by Stance', fontsize=14)
    ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()])

    plt.tight_layout()
    plt.savefig('violin_sentiment_stance_single.png', dpi=200)  # always save

    if show_plot:
        plt.show()  # only show if explicitly requested

    plt.close()  # close figure to prevent lingering plots
    print("Violin plot saved to 'violin_sentiment_stance_single.png'.")

if __name__ == "__main__":
    # Change to True if you want to display the figure interactively
    plot_violin(show_plot=False)
