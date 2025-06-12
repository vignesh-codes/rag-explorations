# RAG Explorations

This repository contains experiments and implementations exploring Retrieval-Augmented Generation (RAG) techniques. RAG is a powerful approach that combines the benefits of retrieval-based and generative AI methods to create more accurate and contextually relevant responses.

## Overview

RAG works by:
1. Retrieving relevant documents/passages from a knowledge base
2. Augmenting the input prompt with this retrieved context
3. Generating responses using the combined context

## Project Structure

The repository will contain:
- Implementation examples of RAG systems
- Experiments with different retrieval methods
- Comparisons of various embedding techniques
- Performance evaluations and benchmarks

## Getting Started

Instructions for setting up and running the examples will be added as they are developed.


To Understand about RAG
- https://www.promptingguide.ai/research/rag
- https://www.youtube.com/watch?v=ea2W8IogX80
- https://www.youtube.com/watch?v=2TJxpyO3ei4

To run the project:
1. Install the dependencies - `pip install -r requirements.txt`
2. Follow this screenshot - `docs\run-program.png`
3. Run the program - `python populate_database.py reset` -> This will reset the database
4. Run the program - `python populate_database.py populate` -> This will populate the database with the documents in the `data` folder
5. Run the program - `python query_data.py "how to win the monopoly game?"` -> This will query the database and return the most relevant documents



## Future Work

This is great that we can use RAG to answer questions about the documents in the database.

I am thinking of using RAG to retrive information of large codebase (some open source projects like kafka, otel, etc) and then use RAG to answer questions about the code in the database. For example - Who is the author of this piece of code? What is the purpose of this piece of code?

### Example queries a contributor or maintainer can run:

"When was the default timeout value last changed in NGINX config?"

"Who changed Kafka partition count and why?"

"What issues were linked to changing memory limits in deployment.yaml?"

"What configs are commonly changed for scaling service X?"

"What was the impact of switching TLS version in config files?"

### The workflow for this will be like this:
Repo Scanner → Git history parser → PR/issues scraper → Document builder → Embedder → Vector DB → Query API/UI



### University program related questions chatbot

- What is the purpose of this program?
- What is the duration of this program?
- What is the eligibility criteria for this program?
- What is the application process for this program?
- What is the selection process for this program?
- What is the duration of this program?

we need to scrape the website and get the data and then use RAG to answer the questions.



