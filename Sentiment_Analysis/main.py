import argparse
import logging
from utils.scraper import scrape_headlines
from utils.classifier import classify_headlines
from utils.analyzer import analyze_sentiment
from utils.utils import configure_logging

def main(url: str, stock_keyword: str = None) -> dict:
    """Main function to orchestrate the scraping and analysis of headlines.
    
    Args:
    url (str): The URL to scrape headlines from.
    stock_keyword (str, optional): The keyword to filter headlines. Defaults to None.

    Returns:
    dict: The sentiment analysis results.
    """
    headlines = scrape_headlines(url)

    if stock_keyword:
        headlines = [headline for headline in headlines if stock_keyword.lower() in headline.lower()]

    sentiment_analysis_results = analyze_sentiment(headlines)
    
    return sentiment_analysis_results

def analyze_results(results: dict) -> dict:
    """Analyzes the sentiment analysis results and prints a summary.

    Args:
    results (dict): The sentiment analysis results.

    Returns:
    dict: A summary of the analysis.
    """
    positive_headlines = 0
    negative_headlines = 0
    cumulative_positive_score = 0.0
    cumulative_negative_score = 0.0

    for headline, sentiment in results.items():
        if sentiment[0]['label'] == 'POSITIVE':
            positive_headlines += 1
            cumulative_positive_score += sentiment[0]['score']
        else:
            negative_headlines += 1
            cumulative_negative_score += sentiment[0]['score']

        logging.info(f"Headline: {headline}")
        logging.info(f"Sentiment: {sentiment[0]['label']}, Score: {sentiment[0]['score']}\n")

    analysis_summary = {
        "Total Positive Headlines": positive_headlines,
        "Cumulative Score for Positive Headlines": cumulative_positive_score,
        "Total Negative Headlines": negative_headlines,
        "Cumulative Score for Negative Headlines": cumulative_negative_score,
    }

    for key, value in analysis_summary.items():
        logging.info(f"{key}: {value}")

    return analysis_summary

if __name__ == "__main__":
    configure_logging()
    
    parser = argparse.ArgumentParser(description="A tool to scrape, analyze, and classify news headlines.")
    parser.add_argument("url", help="The URL to scrape headlines from.")
    parser.add_argument("--stock_keyword", help="The keyword to filter headlines.", default=None)

    args = parser.parse_args()

    results = main(args.url, args.stock_keyword)
    analyze_results(results)
