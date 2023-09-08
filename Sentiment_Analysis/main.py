from scraper import scrape_headlines
from classifier import classify_headlines
from analyzer import analyze_sentiment
from utils import configure_logging

if __name__ == "__main__":
    # Replace with the actual URL of the CNBC news portal you want to analyze
    news_url = "https://www.cnbc.com/world/"
    stock_keyword = None #"AMD"

    results = main(news_url, stock_keyword)

    # Initialize counters
    positive_headlines = 0
    negative_headlines = 0
    cumulative_positive_score = 0.0
    cumulative_negative_score = 0.0

    # Categorize and sum the results
    for headline, sentiment in results:
        if sentiment[0]['label'] == 'POSITIVE':
            positive_headlines += 1
            cumulative_positive_score += sentiment[0]['score']
        else:
            negative_headlines += 1
            cumulative_negative_score += sentiment[0]['score']
        

        print(f"Headline: {headline}")
        print(f"Sentiment: {sentiment[0]['label']}, Score: {sentiment[0]['score']}\n")

    # Print the results
    print(f"Total Positive Headlines: {positive_headlines}")
    print(f"Cumulative Score for Positive Headlines: {cumulative_positive_score}")

    print(f"Total Negative Headlines: {negative_headlines}")
    print(f"Cumulative Score for Negative Headlines: {cumulative_negative_score}")
