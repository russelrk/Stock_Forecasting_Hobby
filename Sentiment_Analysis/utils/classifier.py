from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def classify_headlines(headlines: list[str]) -> list[str]:
    try:
        # Load a pre-trained model for zero-shot classification
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        logging.error(f"Failed to load the zero-shot classification model: {e}")
        return []

    # Define the categories for the zero-shot classifier
    candidate_labels = ["stock news", "non-stock news"]
    
    stock_related_headlines = []
    for headline in headlines:
        try:
            # Perform zero-shot classification
            classification_result = classifier(headline, candidate_labels)
            if classification_result["labels"][0] == "stock news":
                stock_related_headlines.append(headline)
        except Exception as e:
            logging.error(f"Failed to classify headline: {headline}. Error: {e}")

    return stock_related_headlines
