import requests
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_stock_news(api_key: str, symbol: str) -> list[str]:
    """
    Retrieves stock-related news using Alpha Vantage API.

    Args:
    api_key (str): The API key for Alpha Vantage.
    symbol (str): The stock symbol to retrieve news for.

    Returns:
    List[str]: A list of news headlines.
    """

    
    headlines = []

    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()

        news_data = json.loads(response.text)

        # Extracting headlines - adjust the key based on the API response structure
        headlines = [item['headline'] for item in news_data.get('feed', [])]

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to retrieve data from the API. Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred. Error: {e}")

    return headlines

# Example usage
api_key = 'YOUR_API_KEY'  # Replace with your actual Alpha Vantage API key
stock_symbol = 'AAPL'  # Example stock symbol
headlines = get_stock_news(api_key, stock_symbol)
print(headlines)
