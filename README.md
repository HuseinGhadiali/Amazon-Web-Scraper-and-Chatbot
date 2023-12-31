# Amazon Web Scraper and Chatbot

This repository provides a powerful combination of web scraping and conversational AI tools. It houses a web scraper specifically designed to extract data from Amazon's search results based on any provided search term. This data is then stored in a CSV file. Additionally, there's a chatbot that uses this CSV to furnish insightful responses to user queries.

## How It Works

### Web Scraper:
- **Libraries Utilized**: `csv`, `bs4`, and `selenium`.
- **Functionality**:
  - Formulates the search results URL for a specified search term.
  - Navigates to each product's individual page to extract a detailed description.
  - Captures relevant product details such as name, price, rating, number of reviews, and product link from both the search results and individual product pages.
  - Saves the collated data to a CSV file.

### Chatbot:
- **Libraries Utilized**: `streamlit`, `langchain`, and `OpenAI`.
- **Functionality**:
  - Processes the data from the CSV generated by the web scraper.
  - Divides the data text into workable chunks.
  - Engages the OpenAI API to create embeddings and retrieve pertinent data portions.
  - Constructs a comprehensive response to user queries using the data.

## Use Case

This toolkit is especially invaluable for those seeking detailed insights about Amazon products without the manual effort of sifting through countless search results. A simple query, such as "top 5 products based on reviews" or "show all 'Optimum nutrition' products that have atleast 20 gram of protein" enables the chatbot to swiftly deliver precise results, leveraging the pre-stored CSV data. This not only streamlines the user's search process but also ensures the accuracy and relevancy of the results.

## Requirements

- `csv`
- `bs4`
- `selenium`
- `streamlit`
- `langchain`
- `OpenAI`

Ensure the above libraries are installed. Typically, you can get these via `pip`:

```bash
pip install bs4 selenium streamlit langchain OpenAI
```

(Note: Some libraries might require additional dependencies.)

## How to Use

1. Make sure you've scraped the necessary data using the web scraper. This is done by running the script and providing a search term, which will create a CSV file containing detailed information of products related to the search term.
2. Navigate to the repository's root directory in your terminal or command prompt.
3. To launch the chatbot interface, execute the command:

```bash
python -m streamlit run chatbot.py
```

Follow on-screen prompts to start posing questions and receive answers based on the scraped Amazon data!
