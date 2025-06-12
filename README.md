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

## License

This project is open source and available under the MIT License.