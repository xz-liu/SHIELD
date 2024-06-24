import time
from datetime import datetime
from google_books_api_wrapper.api import GoogleBooksAPI
from utils import *

import requests


def get_book_publish_date_google(book_title):
    api_key=os.getenv("GOOGLE_API_KEY")
    # Construct the API request URL
    url = f'https://www.googleapis.com/books/v1/volumes?q={book_title}&key={api_key}'

    # Send the request to the Google Books API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        if 'items' in data:
            # Get the first item from the search results
            book_info = data['items'][0]['volumeInfo']

            # Extract the publish date
            publish_date = book_info.get('publishedDate', 'No publish date found')

            # Parse the publish date to int
            try:
                # Extract the year part from the date string
                year = int(publish_date[:4])
                return year
            except ValueError:
                return 0
        else:
            return 0
    else:
        return 0


def search_titles(title_to_search):
    # Initialize the WebDriver (e.g., for Chrome)

    # Define the base URL
    url = "https://api.publicrecords.copyright.gov/search_service_external/simple_search_dsl"

    # Define the query parameters
    params = {
        "page_number": 1,
        "query": title_to_search,
        "column_name": "title",
        "records_per_page": 10,
        "sort_order": "asc",
        "highlight": "true",
        "model": "",
        'registration_class': 'TX',
        'type_of_query': 'starts_with',
    }

    # Send the GET request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Print the results
        return data
    else:
        print(f"Request failed with status code {response.status_code}")
        return {}



def search_catalog_based_on_title(title):
    url = "https://catalog.archives.gov/api/v2/records/search"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": os.getenv("ARCHIVES_API_KEY")
    }

    params = {
        "title": title
    }

    resp = requests.get(url, headers=headers, params=params)
    return resp.json()


def extract_useful_info(resp):
    """
    Extract useful information from a given JSON data structure.

    Args:
        data (dict): The JSON data from which to extract information.

    Returns:
        dict: A dictionary containing the extracted information.
    """
    useful_info_list = []
    for data in resp.get('data', []):
        useful_info = {
            'title': data.get('hit', {}).get('title_concatenated', 'N/A'),
            'recordation_date': data.get('hit', {}).get('recordation_date', 'N/A'),
            'representative_date': data.get('hit', {}).get('representative_date', 'N/A'),
            'execution_date': data.get('hit', {}).get('execution_date', 'N/A'),
            'copyright_number': data.get('hit', {}).get('copyright_number_for_display', 'N/A'),
            'organizations': [org.get('name_organization_indexed_form', 'N/A') for org in
                              data.get('hit', {}).get('organizations', [])],
            'primary_title': {
                'title': data.get('hit', {}).get('primary_titles_list', [{}])[0].get('title_primary_title_title_proper',
                                                                                     'N/A'),
                'statement_of_responsibility': data.get('hit', {}).get('primary_titles_list', [{}])[0].get(
                    'title_primary_title_statement_of_responsibility', 'N/A'),
                'medium': data.get('hit', {}).get('primary_titles_list', [{}])[0].get('title_primary_title_medium',
                                                                                      'N/A')
            },
            'control_number': data.get('hit', {}).get('control_number', 'N/A'),
            'recordation_number': data.get('hit', {}).get('recordation_number', 'N/A'),
            'general_note': data.get('hit', {}).get('general_note', ['N/A'])[0]
        }
        useful_info_list.append(useful_info)
    return useful_info_list


def search_and_extract_w_cache(data):
    try:
        cached_results = load_from('cached_results.json', backend='json', verbose=False)
    except:
        cached_results = {}
    if data in cached_results:
        useful_info = cached_results[data]
    else:
        resp = search_titles(data)
        useful_info = extract_useful_info(resp)
        cached_results[data] = useful_info
        save_to('cached_results.json', cached_results, backend='json')
    return json.dumps(useful_info)


def search_book_gutenberg(title, cache_file='cache_search_book_gutenberg.json'):
    # Load the cache if it exists
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    # Check if the title is already in cache
    if title in cache:
        return cache[title]

    # Search the book using Gutendex API
    base_url = "https://gutendex.com/books"
    params = {
        'search': title
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        results = response.json()
        if results['results']:
            # Find the book with the highest download count
            top_book = max(results['results'], key=lambda x: x['download_count'])
            book_info = {
                'Title': top_book['title'],
                'Author': ', '.join(author['name'] for author in top_book['authors']),
                'Download count': top_book['download_count'],
                'Copyright status': 'Public domain' if not top_book['copyright'] else 'Protected'
            }
            cache[title] = book_info
        else:
            cache[title] = "No results found."

        # Save the updated cache
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=4)

        return cache[title]
    else:
        return "No results found."


def is_public_domain(publication_year):
    current_year = datetime.now().year

    # 1923年之前出版的作品在公共领域
    if publication_year < 1923:
        return True

    # 1923年到1977年之间出版的作品
    elif 1923 <= publication_year <= 1977:
        if publication_year + 95 < current_year:
            return True
        else:
            return False

    elif publication_year >= 1978:
        if publication_year + 95 < current_year:
            return True
        else:
            return False

    # 如果年份无效
    else:
        raise ValueError("Invalid publication year")


def google_book_search(title, api_key):
    # URL for Google Books API

    get_book_by_title = GoogleBooksAPI().get_book_by_title(title)
    print(get_book_by_title)
    print()
    breakpoint()

def search_book(title):
    '''
    First search Gutenberg for the book title. if Not found, search Public Records.
    '''
    result = search_book_gutenberg(title)
    print('Search result from Gutenberg of title:', title, 'is:', result)
    if result == "No results found.":
        result = search_and_extract_w_cache(title)
        time.sleep(3)
    return result


# Example usage:
if __name__ == "__main__":
    print(google_book_search("A Tale of Two Cities", os.getenv("GOOGLE_API_KEY")))
    # print(search_book_gutenberg("A Tale of Two Cities"))
    # print(search_book_gutenberg("The Da Vinci Code"))
    # print(search_book_gutenberg("Bible"))
