# filename: financial_news_scraper.py
import requests
from bs4 import BeautifulSoup

# Fetch the HTML content of a different financial news website
url = 'https://www.cnbc.com/world/?region=world'
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract the relevant news articles
    news_articles = soup.find_all('a', class_='Card-title')
    for article in news_articles:
        print(article.text)
else:
    print("Failed to retrieve the web page")