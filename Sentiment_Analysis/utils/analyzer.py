from transformers import pipeline
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def analyze_sentiment(headlines: list[str]) -> list[tuple]:
    try:
        # Load the pre-trained BERT model fine-tuned for sentiment analysis
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english", 
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
        )
    except Exception as e:
        logger.error(f"Failed to load the sentiment analysis model: {e}")
        return []
    
    try:
        # Load a pre-trained model for zero-shot classification
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        logger.error(f"Failed to load the headline classification model: {e}")
        return []
    
    sentiment_analysis_results = []
    
    # Define the categories for the zero-shot classifier
    candidate_labels = ["stock news", "non-stock news"]
    
    for headline in headlines:
        # Perform zero-shot classification to determine if the headline is stock-related
        classification_result = classifier(headline, candidate_labels)
        if classification_result["labels"][0] == "stock news":
            try:
                # Perform sentiment analysis
                sentiment_result = sentiment_analyzer(headline)
                sentiment_analysis_results.append((headline, sentiment_result))
            except Exception as e:
                logger.error(f"Failed to analyze sentiment for headline: {headline}. Error: {e}")
    
    return sentiment_analysis_results
