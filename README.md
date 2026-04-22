# Content-Based Fashion Recommendation System
Project Overview
This project is a prototype of a Content-Based Recommendation System designed for e-commerce. 
It analyzes textual product descriptions to find and recommend similar fashion items to users. 
The system is built using Natural Language Processing (NLP) techniques and Machine Learning algorithms to ensure fast and accurate personalization.

The project uses the real-world H&M Personalized Fashion Recommendations dataset.

Features
* Data Optimization: Uses the `Apache Parquet` format for fast and memory-efficient data loading.
* NLP Preprocessing: Cleans and normalizes product descriptions using Regular Expressions (removing punctuation and special characters).
* TF-IDF Vectorization: Converts unstructured text into a sparse mathematical matrix, highlighting unique features and ignoring stop-words.
* Cosine Similarity: Calculates the angle between item vectors to find the closest semantic matches.
* Duplicate Post-Filtering: Ensures diverse output by automatically skipping items with the exact same name (e.g., different sizes or colors of the same model).

Tech Stack
* Language: Python 3
* Data Manipulation: Pandas
* Machine Learning: Scikit-learn
* Data Storage: Apache Parquet (`pyarrow` / `fastparquet`)
