from bs4 import BeautifulSoup
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_headlines(url: str) -> list[str]:
    """
    Scrapes headlines from the provided URL.
    
    Args:
    url (str): The URL to scrape headlines from.
    
    Returns:
    List[str]: A list of headlines.
    """
    headlines = []
    
    try:
        # Validate URL (basic)
        if not url.startswith('http'):
            logger.error("Invalid URL provided.")
            return headlines
        
        # Request the webpage content
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        
        # Parse the webpage content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find and extract headlines (the class name should be verified and updated as per the website structure)
        headlines = [element.get_text() for element in soup.find_all(class_='Card-title')]
        
        if not headlines:
            logger.warning("No headlines found. The website structure might have changed.")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to retrieve the webpage. Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred. Error: {e}")
    
    return headlines

