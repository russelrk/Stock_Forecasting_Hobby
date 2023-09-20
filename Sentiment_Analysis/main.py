import argparse
from utils.scraper import scrape_headlines
from utils.classifier import classify_headlines
from utils.analyzer import analyze_sentiment
from utils.utils import configure_logging

def main(url, stock_keyword=None):
    headlines = scrape_headlines(url)

    if stock_keyword:
        headlines = [headline for headline in headlines if stock_keyword.lower() in headline.lower()]

    sentiment_analysis_results = analyze_sentiment(headlines)

    return sentiment_analysis_results

def analyze_results(results):
    positive_headlines = 0
    negative_headlines = 0
    cumulative_positive_score = 0.0
    cumulative_negative_score = 0.0

    for headline, sentiment in results:
        if sentiment[0]['label'] == 'POSITIVE':
            positive_headlines += 1
            cumulative_positive_score += sentiment[0]['score']
        else:
            negative_headlines += 1
            cumulative_negative_score += sentiment[0]['score']

        print(f"Headline: {headline}")
        print(f"Sentiment: {sentiment[0]['label']}, Score: {sentiment[0]['score']}\n")

    analysis_summary = {
        "Total Positive Headlines": positive_headlines,
        "Cumulative Score for Positive Headlines": cumulative_positive_score,
        "Total Negative Headlines": negative_headlines,
        "Cumulative Score for Negative Headlines": cumulative_negative_score,
    }

    for key, value in analysis_summary.items():
        print(f"{key}: {value}")

    return analysis_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A tool to scrape, analyze, and classify news headlines.")
    parser.add_argument("url", help="The URL to scrape headlines from.")
    parser.add_argument("--stock_keyword", help="The keyword to filter headlines.", default=None)

    args = parser.parse_args()

    results = main(args.url, args.stock_keyword)
    analyze_results(results)
