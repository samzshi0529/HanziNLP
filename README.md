# HanziNLP

An **user-friendly** and **easy-to-use** Natural Language Processing package specifically designed for Chinese text analysis, modeling, and visualization.

## Table of Contents
- [Introduction](#introduction)
  - [Related Links](#related-links)
  - [Installing and Usage](#installing-and-usage)
- [Character and Word Counting](#character-and-word-counting)
- [Font Management](#font-management)
- [Text Segmentation](#text-segmentation)
- [Stopword Management](#stopword-management)
- [Text Representation](#text-representation)
- [Text Similarity](#text-similarity)
- [Word Embeddings](#word-embeddings)
- [Topic Modeling](#topic-modeling)
- [Sentiment Analysis](#sentiment-analysis)

## Developer

To anyone using HanziNLP, big thanks to you from the developer 施展,Samuel Shi! 🎉🎉🎉 

For any improvement and more information about me, you can find via the following ways:
- **Personal Email**: samzshi@sina.com
- **Personal Webiste**: [https://www.samzshi.com/](https://www.samzshi.com/)
- **Linkedin**: [www.linkedin.com/in/zhanshisamuel](www.linkedin.com/in/zhanshisamuel)

## Introduction

Welcome to **HanziNLP** 🌟 - an ready-to-use toolkit for Natural Language Processing (NLP) on Chinese text, while also accommodating English. It Offers a suite of user-friendly tools for various NLP tasks and features an interactive dashboard for dynamic insights into NLP functionalities. Moreover, HanziNLP features an interactive dashboard, providing a dynamic overview and insights into various NLP functionalities. 

### Related Links

- **GitHub Repository**: Explore my code and contribute on [GitHub](https://github.com/samzshi0529/HanziNLP).
- **PyPI Page**: Find me on [PyPI](https://libraries.io/pypi/HanziNLP) and explore more about how to integrate HanziNLP into your projects.

### Installation and Usage

Getting started with HanziNLP is as simple as executing a single command!

```python
pip install HanziNLP
```

## Character and Word Counting 📊

🚀 This basic function count the characters and words in your text, sparing you the manual effot of identifying and splitting Chinese words on your own. 

### char_freq and word_freq Functions
- `char_freq`: Function to calculate the frequency of each character in a given text.
- `word_freq`: Function to calculate the frequency of each word in a given text.
### Code Example
```python
from HanziNLP import char_freq, word_freq

text = "你好, 世界!"
char_count = char_freq(text)
word_count = word_freq(text)

print(f"Character Count: {char_count}")
print(f"Word Count: {word_count}")
```
### Output Example
```python
Charater Count: 4
Word Count: 2
```
## Font Management

When visualizing Chinese text in Python environment, font is a vital resource which is often needed from manual importing. HanziNLP have built-in list of fonts for usage right away. You can use list_fonts() to see and filter all available fonts and use get_font() to retrieve a specific font path for visualization purposes. All built-in fonts are from Google fonts that are licensed under the Open Font License, meaning one can use them in your products & projects – print or digital, commercial or otherwise.

### list_fonts and get_font Functions
- `list_fonts`: List all available fonts.
- `get_font`: Retrieve a specific font for visualization purposes.

#### list_fonts() example
```python
from HanziNLP import list_fonts

# List all available fonts
list_fonts()
```
#### output
![Example Image](README_PIC/list_fonts().png)

#### get_font() example
```python
from HanziNLP import get_font

font_path = get_font('ZCOOLXiaoWei-Regular')
```
#### output
![Example Image](README_PIC/get_font.png)

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
