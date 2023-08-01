'''
Python script to performing concurrent web scraping and text cleaning for a list of URLs. 
It's fetching and processing a list of web pages in parallel to increase the overall speed and efficiency of the operation.
'''

import concurrent.futures
# for parallel processing
import requests
#making HTTP requests in Python
from bs4 import BeautifulSoup
# parsing HTML data
import re
# regular expressions

class WebPage:
    
    def __init__(self, url):
        self.url = url
        
        # Define headers with a User-Agent to mimic a browser
        # In this function, the headers dictionary is created with a "User-Agent" entry that mimics the Mozilla Firefox browser. 
        # When a GET request is made to the URL with requests.get(url, headers=headers), 
        # the server that receives the request will see the User-Agent string "Mozilla/5.0", which is commonly associated with web browsers. 
        # This helps to avoid blocks that some websites might put in place to prevent scraping by bots.
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.raw_text = None
        self.clean_text = None
        
    # Step 1: Web Scraping
    # Function to scrape webpage content
    def scrape(self):
        
        try:
            # Use requests.get() to fetch the webpage content via a url 
            with requests.get(self.url, headers=self.headers) as response:
                # Parse the webpage content using BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
            
                # Remove the footer from the soup object
                footer = soup.find('footer')
                if footer:
                    footer.extract()
                    
                # Modify this line to find the main content of the page
                # This will depend on the structure of the webpage
                # Here, we're getting all the text inside <p>, <h1>, <h2>, etc., tags
                text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table'])
                
                # Join all the text elements into a single string
                text = ' '.join([elem.get_text() for elem in text_elements])
                
                # Finding image captions/descriptions
                # Using the 'alt' attribute of the <img> tag
                img_elements = soup.find_all('img')
                img_descriptions = ' '.join([img.get('alt', '') for img in img_elements if img.get('alt')])

                # Using <figcaption> tag
                figcaption_elements = soup.find_all('figcaption')
                figcaptions = ' '.join([fig.get_text() for fig in figcaption_elements])

                # Combine the raw text, image descriptions, and figcaptions
                self.raw_text = f'{text} {img_descriptions} {figcaptions}'
                
                # OR
                # Just extract all the text from the parsed HTML, including headers and footers
                #raw_text = soup.get_text()
        
        # if any RequestException is raised during the requests.get() call,
        # Python will catch it and assign it to the variable e
        except requests.exceptions.RequestException as e:
            # print the URL being scraped and a description of the exception (e)
            print(f"Error scraping {self.url}: {e}")
            self.raw_text = None
     

    # Step 2: Text Cleaning - minimal for FinBERT since it is capable of handling raw text data
    def clean(self):
        if self.raw_text is not None:
            try:
                # Removing any HTML artifacts using regular expressions
                text = re.sub('<.*?>', '', self.raw_text)

                # Remove references if any, like [1], [2] etc.
                text = re.sub(r'\[[0-9]*\]', ' ', text)
                # Replace multiple spaces with a single space
                text = re.sub(r'\s+', ' ', text)
                
                # to remove words greater than 10 characters
                # Split the text into words
                words = text.split()
                # Use a list comprehension to filter out long words
                short_words = [word for word in words if len(word) <= 10]
                # Join the short words back into a string
                self.clean_text = ' '.join(short_words)
                
                # OR 
                # Truncate the text to remove last 1000 characters as it is extra
                # clean_text = clean_text[:-1000]
            
            # if any RequestException is raised during text cleaning,
            # Python will catch it and assign it to the variable 'e'
            except Exception as e:
                # print the URL being scraped and a description of the exception (e)
                print(f"Error cleaning the text of {self.url}: {e}")
                self.clean_text = None


class WebPageScraper:
    def __init__(self, urls):
        self.urls = urls
        
    def scrape_all(self):
        '''
        The scrape_all() method orchestrates the scraping and cleaning of webpages given a list of URLs. 
        It does so by creating a WebPage object for each URL, concurrently scraping these webpages, cleaning the scraped data, 
        and returning a list of cleaned text in the order of the original URLs.
        '''
        
        # Create a list of WebPage objects for each URL in the provided list of URLs
        webpages = [WebPage(url) for url in self.urls]
        
        # Create a ThreadPoolExecutor
        # Download and clean all pages in parallel using a pool of worker threads to execute calls asynchronously
        with concurrent.futures.ThreadPoolExecutor() as executor:
            
            # executor.map() takes a function and an iterable and applies the function to every item in the iterable
            # in this case, it's using a lambda function to call the 'scrape' method on each 'WebPage' object in 'webpages'
            # this is done concurrently, meaning it's done in parallel using multiple threads
            executor.map(lambda webpage: webpage.scrape(), webpages)
        
        # Loop over the list of 'WebPage' objects
        for webpage in webpages:
            # Call the 'clean' method on each 'WebPage' object to clean the scraped text
            webpage.clean()
        
        # Finally, return a list of the cleaned text from each 'WebPage' object
        # The order of the cleaned text will be the same as the original order of the URLs
        return [webpage.clean_text for webpage in webpages]
