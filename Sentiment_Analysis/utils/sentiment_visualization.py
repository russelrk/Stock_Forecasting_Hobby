import matplotlib.pyplot as plt
import seaborn as sns

def visualize_sentiment_results(results):
    """
    Visualizes the sentiment analysis results using a bar plot.

    Parameters:
    results (list[tuple]): A list of tuples containing headlines and sentiment analysis results.

    Returns:
    None
    """
    if not results:
        print("No results to visualize.")
        return

    # Extract sentiment scores and labels
    labels = [res[0] for res in results]
    scores = [res[1][0]['score'] if res[1][0]['label'] == 'POSITIVE' else -res[1][0]['score'] for res in results]

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Create a bar plot for the sentiment scores
    sns.barplot(x=scores, y=labels, palette='viridis')

    # Set plot labels and title
    plt.xlabel('Sentiment Score')
    plt.ylabel('Headlines')
    plt.title('Sentiment Analysis of Headlines')

    # Display the plot
    plt.show()
