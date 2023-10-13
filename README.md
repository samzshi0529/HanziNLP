# HanziNLP

A comprehensive Natural Language Processing package specifically designed for Chinese text.

## Table of Contents
- [Introduction](#introduction)
- [Character and Word Counting](#character-and-word-counting)
- [Font Management](#font-management)
- [Text Segmentation](#text-segmentation)
- [Stopword Management](#stopword-management)
- [Text Representation](#text-representation)
- [Text Similarity](#text-similarity)
- [Word Embeddings](#word-embeddings)
- [Topic Modeling](#topic-modeling)
- [Sentiment Analysis](#sentiment-analysis)

## Introduction

HanziNLP is a specialized NLP package tailored for handling Chinese text. It provides a suite of tools to perform various NLP tasks, from basic text preprocessing to advanced text analysis and modeling.

## Character and Word Counting

### char_freq and word_freq Functions
- `char_freq`: Function to calculate the frequency of each character in a given text.
- `word_freq`: Function to calculate the frequency of each word in a given text.

## Font Management

### list_fonts and get_font Functions
- `list_fonts`: List all available fonts.
- `get_font`: Retrieve a specific font for visualization purposes.

## Text Segmentation

### sentence_segment and word_tokenize Functions
- `sentence_segment`: Segment the input text into sentences.
- `word_tokenize`: Tokenize the input text into words and remove stopwords.

## Stopword Management

### list_stopwords and load_stopwords Functions
- `list_stopwords`: List all available stopwords.
- `load_stopwords`: Load stopwords from a specified file.

## Text Representation

### BoW, ngrams, TF_IDF, and TT_matrix Functions
- `BoW`: Generate a Bag of Words representation of the input text.
- `ngrams`: Generate n-grams from the input text.
- `TF_IDF`: Generate a TF-IDF representation of the input text.
- `TT_matrix`: Generate a term-term matrix of the input text.

## Text Similarity

### text_similarity Function
- `text_similarity`: Calculate the similarity between two texts using various methods.

## Word Embeddings

### Word2Vec and get_bert_embeddings Functions
- `Word2Vec`: Obtain word embeddings using the FastText model.
- `get_bert_embeddings`: Obtain word embeddings using the BERT model.

## Topic Modeling

### lda_model and print_topics Functions
- `lda_model`: Train an LDA model on the input text.
- `print_topics`: Print the topics identified by the LDA model.

## Sentiment Analysis

### sentiment Function
- `sentiment`: Perform sentiment analysis on the input text using a specified pre-trained model.

---

Feel free to modify the descriptions and add any additional information or usage examples that you think would be helpful for users of your package!
