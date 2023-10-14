# HanziNLP

一个**用户友好**且**易于使用**的自然语言处理包，专为中文文本分析、建模和可视化而设计。HanziNLP中的所有功能都支持中文文本，并且非常适用于中文文本分析！

<detail>
<summary>🇨🇳 Chinese Version (点击查看中文版本)</summary>
## 目录
- [1. 快速开始](#1-快速开始)
  - [1.1 相关链接](#11-相关链接)
  - [1.2 安装和使用](#12-安装和使用)
  - [1.3 交互式仪表板](#13-交互式仪表板)

<detail>

# HanziNLP

An **user-friendly** and **easy-to-use** Natural Language Processing package specifically designed for Chinese text analysis, modeling, and visualization. All functions in HanziNLP supports Chinese text and works well for Chinese text!

## Table of Contents
- [1. Quick Start](#1-quick-start)
  - [1.1 Related Links](#11-related-links)
  - [1.2 Installing and Usage](#12-installing-and-usage)
  - [1.3 Interactive Dashboard](#13-interactive-dashboard)
- [2. Character and Word Counting](#2-character-and-word-counting)
- [3. Font Management](#3-font-management)
- [4. Text Segmentation](#4-text-segmentation)
  - [4.1 Stopword Management](#41-stopword-management)
  - [4.2 Sentence Segmentation](#42-sentence-segmentation)
  - [4.3 Word Tokenization](#43-word-tokenization)
- [5. Text Representation](#5-text-representation)
  - [5.1 BoW (Bag of Words)](#51-bow-bag-of-words)
  - [5.2 ngrams](#52-ngrams)
  - [5.3 TF_IDF (Term Frequency-Inverse Document Frequency)](#53-tf_idf-term-frequency-inverse-document-frequency)
  - [5.4 TT_matrix (Term-Term Matrix)](#54-tt_matrix-term-term-matrix)
- [6. Text Similarity](#6-text-similarity)
- [7. Word Embeddings](#7-word-embeddings)
  - [7.1 Word2Vec](#71-word2vec)
  - [7.2 BERT Embeddings](#72-bert-embeddings)
- [8. Topic Modeling](#8-topic-modeling)
  - [8.1 Latent Dirichlet Allocation (LDA) model](#81-latent-dirichlet-allocation-lda-model)
  - [8.2 LDA print_topics function](#82-lda-print-topics-function)
- [9. Sentiment Analysis](#9-sentiment-analysis)

## Developer Note:

To anyone using HanziNLP, big thanks to you from the developer 施展,Samuel Shi! 🎉🎉🎉 

For any improvement and more information about me, you can find via the following ways:
- **Personal Email**: samzshi@sina.com
- **Personal Webiste**: [https://www.samzshi.com/](https://www.samzshi.com/)
- **Linkedin**: [www.linkedin.com/in/zhanshisamuel](www.linkedin.com/in/zhanshisamuel)

## 1. Quick Start

Welcome to **HanziNLP** 🌟 - an ready-to-use toolkit for Natural Language Processing (NLP) on Chinese text, while also accommodating English. It is designed to be user-friendly and simplified tool even for freshmen in python. 

Moreover, HanziNLP features an interactive dashboard for dynamic insights into NLP functionalities, providing a dynamic overview and insights into various NLP functionalities.

### 1.1 Related Links

- **GitHub Repository**: Explore my code and contribute on [GitHub](https://github.com/samzshi0529/HanziNLP).
- **PyPI Page**: Find me on [PyPI](https://libraries.io/pypi/HanziNLP) and explore more about how to integrate HanziNLP into your projects.

### 1.2 Installing and Usage

Getting started with HanziNLP is as simple as executing a single command!

```python
pip install HanziNLP
```

### 1.3 Interactive Dashboard

![Alt Text](README_PIC/dashboard_video.gif)

#### Use the dashboard() by a simple line!

```python
from HanziNLP import dashboard
dashboard()
```

- **Function**: `dashboard()`
- **Purpose**: Present a user-friendly dashboard that facilitates interactive text analysis and sentiment classification, enabling users to observe the impacts of various pre-trained models and tokenization parameters on the processed text and thereby select the optimal model and parameters for their use case.
- **Parameters**: No parameters are required.
- **Returns**: No return value; the function outputs a dashboard interface.

#### Overview

The `dashboard` function introduces a user-interactive dashboard, designed to perform text analysis and sentiment classification, providing users with a hands-on experience to explore and understand the effects of different pre-trained models and tokenization parameters on text processing.

- **Interactive Text Analysis**: Users can input text, observe various text statistics, such as word count, character count, and sentence count, and visualize token frequencies and sentiment classification results.
- **Model Exploration**: Users have the option to specify a classification model from Hugging Face. If left blank, a default model, 'uer/roberta-base-finetuned-chinanews-chinese', is utilized. More about this model can be found on [Hugging Face](https://huggingface.co/uer/roberta-base-finetuned-chinanews-chinese).
- **Tokenization Parameter Tuning**: Users can adjust tokenization settings, such as the 'Jieba Mode' parameter and stopwords selection, and observe the resultant tokens and their respective frequencies.
- **Visualization**: The dashboard provides visual insights into text statistics, word frequencies, and sentiment classification, aiding users in understanding the text analysis results.
- **Sentiment Classification**: The dashboard performs sentiment classification using the specified (or default) model and displays the probability distribution across sentiment labels.

#### Highlight

The `dashboard` function emphasizes **user engagement** and **exploration**. It allows users to interactively engage with various pre-trained models and tokenization parameters, observing their effects on text analysis and sentiment classification. This interactive exploration enables users to make informed decisions, selecting the model and parameters that best align with their specific use case, thereby enhancing their text analysis and natural language processing (NLP) tasks.

## 2. Character and Word Counting

🚀 This basic function count the characters and words in your text, sparing you the manual effot of identifying and splitting Chinese words on your own. 

### char_freq and word_freq Functions
- `char_freq(text, text_only=True)`: Function to calculate the frequency of each character in a given text; If text_only == True, only Chinese and English characters will be counted. If text_only == False, all characters will be counted. Default to be True.
- `word_freq(text)`: Function to calculate the frequency of each word in a given text.
### Example
```python
from HanziNLP import char_freq, word_freq

text = "你好, 世界!"
char_count = char_freq(text)
word_count = word_freq(text)

print(f"Character Count: {char_count}")
print(f"Word Count: {word_count}")
```
### Output 
```python
Charater Count: 4
Word Count: 2
```
## 3. Font Management

When visualizing Chinese text in Python environment, font is a vital resource which is often needed from manual importing. HanziNLP have built-in list of fonts for usage right away. You can use list_fonts() to see and filter all available fonts and use get_font() to retrieve a specific font path for visualization purposes. All built-in fonts are from Google fonts that are licensed under the Open Font License, meaning one can use them in your products & projects – print or digital, commercial or otherwise.

### list_fonts and get_font Functions
- `list_fonts()`: List all available fonts.
- `get_font(font_name, show=True)`: Retrieve a specific font for visualization purposes. If show == True, a sample visualization of the font will be shown. If show == False, nothing will be shown. Default set to be True.

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

font_path = get_font('ZCOOLXiaoWei-Regular') #Enter the font_name you like in list_fonts()
```
#### output
![Example Image](README_PIC/get_font.png)

#### WordCloud Example
You can use the Chinese font_path you defined to make all kinds of plots. A wordcloud example is provided below:
```python
from PIL import Image
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt

# A sample text generated by GPT-4 
text = '在明媚的春天里，小花猫咪悠闲地躺在窗台上，享受着温暖的阳光。她的眼睛闪烁着好奇的光芒，时不时地观察着窗外忙碌的小鸟和蝴蝶。小猫的尾巴轻轻摇动，表达着她内心的舒适和满足。在她的身边，一盆盛开的紫罗兰散发着淡淡的香气，给这个宁静的午后增添了几分诗意。小花猫咪偶尔会闭上她的眼睛，沉浸在这美好的时光中，仿佛整个世界都变得温馨和谐。窗外的樱花树在微风中轻轻摇曳，洒下一片片粉色的花瓣，如梦如幻。在这样的一个悠托的春日里，一切都显得如此美好和平静。'

text = " ".join(text)

# Generate the word cloud
wordcloud = WordCloud(font_path= font_path, width=800, height=800,
                      background_color='white',
                      min_font_size=10).generate(text)

# Display the word cloud
plt.figure(figsize=(5, 5), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.title("sample wordcloud")

plt.show()
```
#### output
![Example Image](README_PIC/wordcloud.png)

## 4. Text Segmentation
Text Segmentatino is a vital step in any NLP tasks. The general step is to segment the sentences, remove stopwords, and tokenize each sentences separately. The detailed instructions are introduced below. 

### 4.1 Stopword Management
To remove stopwords in Chinese text, the package have built-in common stopwords lists include the following ones: (Some stopwords are from [stopwords](https://github.com/goto456/stopwords/))

| Stopword List | File Name |
|----------|----------|
| 中文停用词表 | cn_stopwords.txt |
| 哈工大停用词表 | hit_stopwords.txt |
| 百度停用词表 | baidu_stopwords.txt |
| 四川大学机器智能实验室停用词表 | scu_stopwords.txt |
| 常用停用词表 | common_stopwords.txt |

#### list_stopwords and load_stopwords Functions
- `list_stopwords()`: List all available stopwords.
- `load_stopwords(file_name)`: Load stopwords from a specified file to a list of words. You can then see and use the stopwords for later usage. 

##### list_stopwords example
```python
from HanziNLP import list_stopwords

list_stopwords()
```
##### output 
![Example Image](README_PIC/list_stopwords.png)

##### load_stopwords example
```python
from HanziNLP import load_stopwords

stopwords = load_stopwords('common_stopwords.txt') # Enter the txt file name here
```
##### output 
```python
{'然而',
 'whoever',
 '只限',
 '的确',
 '要不然',
 'each',
 '仍旧',
 '这么点儿',
 '冒',
 '如果',
 '比及',
 '以期',
 '犹自'.....
}
```

### 4.2 Sentence Segmentation
This function segments a whole document or paragraphs into sentences. Support both Chinese and English text.
- `sentence_segment(text)`: Segment the input text into sentences. 

#### sentence_segment example: This example intentially chooses a hard sentence to split.
```python
from HanziNLP import sentence_segment

sample_sentence = 'hello world! This is Sam.。 除非你不说。我今天就会很开心,hello .you。'
sentence_segment(sample_sentence)
```
#### output 
```python
['hello world!', 'This is Sam.', '。', '除非你不说。', '我今天就会很开心,hello .', 'you。']
```

### 4.3 Word Tokenization
As one of the most important step in preprocessing text for NLP tasks, the word_tokenize() function provide a direct way to transform raw Chinese text into tokens. 

- **Function**: `word_tokenize(text, mode='precise', stopwords='common_stopwords.txt', text_only=False, include_numbers=True, custom_stopwords=None, exclude_default_stopwords=False)`
- **Purpose**: Tokenize the input text into words while providing options to manage stopwords effectively.
  
#### Parameters:
- `text` (str): The input Chinese text.
- `mode` (str, optional): Tokenization mode, choose from 'all', 'precise', or 'search_engine'. Default is 'precise'.
- `stopwords` (str, optional): A filename string indicating the stopwords file to be used. Default is 'common_stopwords.txt'.
- `text_only` (bool, optional): If True, only tokenize English and Chinese texts. Default is False.
- `include_numbers` (bool, optional): Include numbers in the tokenized output if True. Default is True.
- `custom_stopwords` (list of str, optional): A list of custom stopwords to be removed. Default is None.
- `exclude_default_stopwords` (bool, optional): Exclude default stopwords if True. Default is False.

#### Returns:
- `list`: A list of tokens, with stopwords removed according to the specified parameters.

#### Example 1：
```python
from HanziNLP import word_tokenize
 
sample = '除非你不说，我今天就会很开心,hello you#$@#@*' # A text intentionally to be hard for tokenization
token = sz.word_tokenize(sample, mode='precise', stopwords='baidu_stopwords.txt', text_only=False, 
                  include_numbers=True, custom_stopwords=None, exclude_default_stopwords=False)
token
```
#### output 
```python
['不', '说', '，', '会', '很', '开心', ',', '#', '$', '@', '#', '@', '*']
```
#### Example 2： set text_only to be True and custom_stopwords to be ['开心']
```python
from HanziNLP import word_tokenize

sample = '除非你不说，我今天就会很开心,hello you#$@#@*'# A text intentionally to be hard for tokenization
token = sz.word_tokenize(sample, mode='precise', stopwords='baidu_stopwords.txt', text_only=True, 
                  include_numbers=True, custom_stopwords=['开心'], exclude_default_stopwords=False)
token
```
#### output: Special characters and the word '开心' are removed
```python
['不', '说', '会', '很']
```

## 5. Text Representation
Building text feature map is the starting point for various Machine Learning or Deep Learning tasks. HanziNLP has incorporate the common feature map methods that can be easily implemented.

### 5.1 BoW (Bag of Words)

- **Function**: `BoW(segmented_text_list)`
- **Purpose**: Generate a Bag of Words representation from a list of segmented texts.
- **Parameters**:
  - `segmented_text_list` (list of str): A list containing segmented texts.
- **Returns**: 
  - `dict`: A dictionary representing word frequencies.

#### example
```python
from HanziNLP import word_tokenize, BoW

sample_sentence = 'hello world! This is Sam.。 除非你不说。我今天就会很开心,hello .you。'
token = word_tokenize(sample_sentence, text_only = True)
bow = BoW(token)
bow
```
#### output 
```python
{'hello': 2, 'world': 1, 'This': 1, 'Sam': 1, '说': 1, '今天': 1, '会': 1, '开心': 1}
```

### 5.2 ngrams

- **Function**: `ngrams(tokens, n=3)`
- **Purpose**: Create and count the frequency of n-grams from a list of tokens.
- **Parameters**:
  - `tokens` (list): A list of tokens.
  - `n` (int, optional): The number for n-grams. Default is 3 (trigrams).
- **Returns**: 
  - `dict`: A dictionary with n-grams as keys and their frequencies as values.

#### example
```python
from HanziNLP import word_tokenize, ngrams

sample_sentence = 'hello world! This is Sam.。 除非你不说。我今天就会很开心,hello .you。'
token = word_tokenize(sample_sentence, text_only = True)
ngram = ngrams(token, n =3)
ngram
```
#### output 
```python
{'hello world This': 1,
 'world This Sam': 1,
 'This Sam 说': 1,
 'Sam 说 今天': 1,
 '说 今天 会': 1,
 '今天 会 开心': 1,
 '会 开心 hello': 1}
```

### 5.3 TF_IDF (Term Frequency-Inverse Document Frequency)

- **Function**: `TF_IDF(text_list, max_features=None, output_format='sparse')`
- **Purpose**: Transform a list of texts into a TF-IDF representation.
- **Parameters**:
  - `text_list` (list of str): A list of tokens to be transformed.
  - `max_features` (int, optional): Maximum number of features (terms) to be extracted. Defaults to None (all features).
  - `output_format` (str, optional): Format of the output matrix ('sparse', 'dense', or 'dataframe'). Defaults to 'sparse'.
- **Returns**: 
  - `matrix`: TF-IDF matrix in the specified format.
  - `feature_names`: List of feature names.

#### example
```python
from HanziNLP import word_tokenize, TF_IDF

sample_sentence = 'hello world! This is Sam.。 除非你不说。我今天就会很开心,hello .you。'
token = word_tokenize(sample_sentence, text_only = True)
tfidf_matrix, feature_names = sz.TF_IDF(token, output_format = 'dataframe')
tfidf_matrix
```
#### output 
![Example Image](README_PIC/TFIDF.png)

### 5.4 TT_matrix (Term-Term Matrix)

- **Function**: `TT_matrix(tokenized_texts, window_size=1)`
- **Purpose**: Generate a term-term matrix from a list of tokenized texts, representing term co-occurrences within a specified window.
- **Parameters**:
  - `tokenized_texts` (list of list of str): A list of tokenized texts.
  - `window_size` (int): The window size for co-occurrence. Default is 1.
- **Returns**: 
  - `np.array`: A square matrix where entry (i, j) is the co-occurrence between term i and term j.
  - `index_to_term`: A dictionary mapping from index to term.

#### example
```python
from HanziNLP import word_tokenize, TT_matrix

sample_sentence = 'hello world! This is Sam.。 除非你不说。我今天就会很开心,hello .you。'
token = word_tokenize(sample_sentence, text_only = True)
matrix, index_to_term = TT_matrix(token, window_size = 1)
matrix
```
#### output 
``` python
array([[0., 4., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [4., 0., 0., 0., 0., 0., 0., 2., 2., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [4., 0., 4., 4., 0., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 4., 0., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 2., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 2., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        2.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.,
        0.]])
```

## 6. Text Similarity

### text_similarity Function
- **Function**: `text_similarity(text1, text2, method='cosine')`
- **Purpose**: To calculate and return the similarity score between two input texts, utilizing a specified method.
- **Parameters**:
  - `text1` (str): The first text string for comparison.
  - `text2` (str): The second text string for comparison.
  - `method` (str): The method utilized for computing similarity. Options include 'cosine', 'jaccard', 'euclidean', or 'levenshtein'. Default is 'cosine'.
- **Returns**: 
  - `float`: A numerical value representing the similarity score between `text1` and `text2`.

#### Overview

The `text_similarity` function is meticulously crafted to calculate the similarity between two text strings, namely `text1` and `text2`, using a method specified by the user. Initially, the function tokenizes the input texts and converts them into vectorized forms. Subsequently, it computes the similarity score based on the chosen method, which can be one of the following: 'cosine', 'jaccard', 'euclidean', or 'levenshtein'.

- **Cosine Similarity**: Measures the cosine of the angle between two non-zero vectors, providing a measure of the cosine of the angle between them.
- **Jaccard Similarity**: Calculates the size of the intersection divided by the size of the union of the two text strings.
- **Euclidean Similarity**: Utilizes the Euclidean distance between two vectors to compute similarity.
- **Levenshtein Similarity**: Employs the Levenshtein distance, or "edit distance", between two strings, normalized to a similarity score.

#### example 1: Jaccard Similarity
```python
from HanziNLP import text_similarity

sample='你好世界'
sample1 = '你好世界，hello world'
text_similarity(sample, sample1, method = 'jaccard')
```
#### output 
```python
0.5
```

#### example 1: Levenshtein Similarity
```python
from HanziNLP import text_similarity

sample='你好世界'
sample1 = '你好世界，hello world'
text_similarity(sample, sample1, method = 'levenshtein')
```
#### output 
```python
0.07692307692307693
```

## 7. Word Embeddings

### 7.1 Word2Vec 
- `Word2Vec`: Obtain word embeddings using the FastText model.
- **Function**: `Word2Vec(text, dimension=300)`
- **Purpose**: Obtain word embeddings for a text that may contain both English and Chinese words, utilizing pre-trained FastText models.
- **Parameters**:
  - `text` (str): The input text, which may encompass both English and Chinese words.
  - `dimension` (int): The dimensionality of the resulting word embeddings. Default is 300.
- **Returns**: 
  - `list of numpy.ndarray`: A list containing the word embeddings for each word present in the input text.

#### Overview

The `Word2Vec` function is designed to generate word embeddings for a given text, which may contain words from both English and Chinese languages, using pre-trained FastText models. Initially, the function checks and downloads the FastText models for English and Chinese if they are not already downloaded. Subsequently, it loads the models and, if requested, reduces their dimensionality to the specified size.

The text is tokenized into words, and for each word, the function checks whether it contains Chinese characters. If a word contains Chinese characters, the Chinese FastText model is used to get its embedding; otherwise, the English model is used. The resulting embeddings are appended to a list which is then returned.

- **Utilizing FastText**: FastText models, which are pre-trained on a large corpus of text, are employed to generate word embeddings.
- **Support for Multiple Languages**: Specifically designed to handle texts containing both English and Chinese words by utilizing respective language models.
- **Dimensionality Reduction**: Offers the flexibility to reduce the dimensionality of the embeddings if a smaller size is desired.

#### example
```python
from HanziNLP import Word2Vec

sample_sentence = 'hello world! This is Sam.。 除非你不说。我今天就会很开心,hello .you。'
result = Word2Vec(sample_sentence)
```

### 7.2 BERT Embeddings
- **Function**: `get_bert_embeddings(text, model="bert-base-chinese")`
- **Purpose**: Retrieve BERT embeddings for a specified text using a pre-trained Chinese BERT model.
- **Parameters**:
  - `text` (str): The input text for which embeddings are to be generated.
  - `model` (str): The name of the pre-trained Chinese BERT model to be utilized. Default is "bert-base-chinese."
- **Returns**: 
  - `sentence_embedding` (list): The sentence embedding represented as a list of floats.
  - `tokens` (list): The tokens associated with the sentence embedding.

#### Overview

The `get_bert_embeddings` function is engineered to extract BERT embeddings for a given text using a specified pre-trained Chinese BERT model. Initially, the function loads the designated BERT model and its corresponding tokenizer. The input text is tokenized and prepared for the model, ensuring it is truncated to a maximum length of 512 tokens to be compatible with the BERT model.

Subsequent to tokenization, the model generates predictions, and the last hidden states of the BERT model are retrieved. The sentence embedding is computed by taking the mean of the last hidden states and converting it to a list of floats. Additionally, the tokens associated with the sentence embedding are obtained by converting the input IDs back to tokens.

- **Utilizing BERT**: Leverages a pre-trained BERT model, renowned for its effectiveness in generating contextual embeddings.
- **Support for Chinese Text**: Specifically tailored to handle Chinese text by utilizing a Chinese BERT model.
- **Token Handling**: Ensures tokens are appropriately managed and returned alongside embeddings for reference and further analysis.

#### example
```python
from HanziNLP import get_bert_embeddings

embeddings, tokens = get_bert_embeddings(text, model = "bert-base-chinese") # enter the BERT Model name you wish to use from Hugging Face
print(f"Tokens: {tokens}")
print(f"Embeddings: {embeddings}")
```

## 8. Topic Modeling
HanziNLP have integrated code to easily implement LDA model to extract topics from large amount of text. More models will be updated: 

### 8.1 Latent Dirichlet Allocation (LDA) model

- **Function**: `lda_model(texts, num_topics=10, passes=15, dictionary=None)`
- **Purpose**: Train a Latent Dirichlet Allocation (LDA) model on the provided texts to extract and identify topics.
- **Parameters**:
  - `texts` (list of list of str): A list of documents, with each document represented as a list of tokens.
  - `num_topics` (int): The number of topics to extract. Default is 10.
  - `passes` (int): The number of training passes through the corpus. Default is 15.
  - `dictionary` (corpora.Dictionary, optional): An optional precomputed Gensim dictionary.
- **Returns**: 
  - `lda_model`: The trained LDA model.
  - `corpus`: The corpus used to train the model.
  - `dictionary`: The dictionary used to train the model.

#### Overview

The `lda_model` function is devised to train an LDA model on a collection of texts, facilitating the extraction and identification of underlying topics. If no precomputed dictionary is provided, the function generates a new one from the input texts. The texts are converted into a bag-of-words representation, and the LDA model is trained using specified or default parameters. The trained model, corpus, and dictionary are returned, enabling further analysis and topic visualization.

- **Topic Modeling**: Utilizes LDA, a popular topic modeling technique, to uncover latent topics in the text data.
- **Flexible Training**: Allows specification of the number of topics, training passes, and optionally, a precomputed dictionary.
- **Applicability**: Suitable for analyzing large volumes of text data to discover thematic structures.

### 8.2 LDA print topics function

- **Function**: `print_topics(lda_model, num_words=10)`
- **Purpose**: Display the top words associated with each topic from a trained LDA model.
- **Parameters**:
  - `lda_model`: The trained LDA model.
  - `num_words` (int): The number of top words to display for each topic. Default is 10.
- **Returns**: 
  - None (Outputs are printed to the console).

#### Overview

The `print_topics` function is designed to display the top words associated with each topic from a trained LDA model, providing a quick and insightful overview of the thematic essence of each topic. By iterating through each topic, it prints the topic index and the top words, aiding in the interpretability and analysis of the topics extracted by the LDA model.

- **Topic Interpretation**: Facilitates easy interpretation of the topics generated by the LDA model.
- **Customizable Output**: Allows the user to specify the number of top words to be displayed for each topic.
- **Insightful Overview**: Provides a succinct and informative overview of the primary themes present in the text data.

#### example
```python
from HanziNLP import sentence_segment, word_tokenize, lda_model, print_topics

sample_sentence = 'hello world! This is Sam.。 除非你不说。我今天就会很开心,hello .you。'
sentences = sentence_segment(sample_sentence)
tokenized_texts = [sz.word_tokenize(sentence) for sentence in sentences]
lda_model, corpus, dictionary = lda_model(tokenized_texts, num_topics=5)
print_topics(lda_model)
```
#### output
``` python
Topic: 0 
Words: 0.231*"This" + 0.231*"Sam" + 0.231*"." + 0.038*"说" + 0.038*"hello" + 0.038*"world" + 0.038*"!" + 0.038*"今天" + 0.038*"开心" + 0.038*"会"
Topic: 1 
Words: 0.231*"world" + 0.231*"!" + 0.231*"hello" + 0.038*"说" + 0.038*"." + 0.038*"Sam" + 0.038*"This" + 0.038*"今天" + 0.038*"会" + 0.038*"开心"
Topic: 2 
Words: 0.091*"说" + 0.091*"This" + 0.091*"!" + 0.091*"hello" + 0.091*"." + 0.091*"world" + 0.091*"Sam" + 0.091*"开心" + 0.091*"今天" + 0.091*"会"
Topic: 3 
Words: 0.146*"." + 0.146*"hello" + 0.146*"," + 0.146*"会" + 0.146*"开心" + 0.146*"今天" + 0.024*"说" + 0.024*"Sam" + 0.024*"!" + 0.024*"world"
Topic: 4 
Words: 0.375*"说" + 0.063*"hello" + 0.063*"." + 0.063*"!" + 0.063*"Sam" + 0.063*"world" + 0.063*"This" + 0.063*"今天" + 0.063*"会" + 0.063*"开心"
```

## 9. Sentiment Analysis
Sentiment Analysis is common in NLP tasks when sentiment of text could contribute to analysis in further research. While there are many ways to do sentiment analysis like using a sentiment dictionary, HanziNLP integrate the function to allow easily using pretrained BERT models or other language models on Huggin Face for text classification. 

### sentiment Function

- **Function**: `sentiment(text, model='hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2', print_all=True, show=False)`
- **Purpose**: Execute sentiment analysis on the input text utilizing the specified pre-trained model and optionally visualize the probability distribution across sentiment labels.
- **Parameters**:
  - `text` (str): The input text subject to sentiment analysis.
  - `model` (str): The identifier of the pre-trained model to be used. You can use any model on **Hugging Face** and copy the model name here to use it to classify the text. Default is 'hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2'.
  - `print_all` (bool): Indicator whether to print probabilities for all labels or only the label with the highest probability. Default is True.
  - `show` (bool): Indicator whether to display a bar chart showing the probability distribution across labels. Default is False.
- **Returns**: 
  - `dict` or `tuple`: If `print_all` is True, a dictionary containing sentiment labels and their corresponding probabilities. If `print_all` is False, a tuple containing the label with the highest probability and its corresponding probability.

#### Overview

The `sentiment` function is tailored to perform sentiment analysis on a provided text using a specified pre-trained model. Upon loading the tokenizer and model, the input text is tokenized and passed through the model to obtain output logits. These logits are then converted to probabilities using the softmax function. The labels corresponding to these probabilities are retrieved from the model’s configuration and stored in a dictionary along with their respective probabilities.

If `show` is set to True, a bar chart visualizing the probability distribution across sentiment labels is displayed. The function returns either a dictionary of all sentiment labels and their corresponding probabilities (if `print_all` is True) or a tuple containing the label with the highest probability and its corresponding probability (if `print_all` is False).

- **Sentiment Analysis**: Utilizes a specified pre-trained model to analyze the sentiment of the input text.
- **Visualization**: Optionally visualizes the probability distribution across sentiment labels using a bar chart.
- **Flexible Output**: Provides flexibility in output, allowing for detailed or concise sentiment analysis results.

#### example
```python
from HanziNLP import sentiment

text = "这个小兄弟弹的太好了"
sentiment= sentiment(text, model = 'touch20032003/xuyuan-trial-sentiment-bert-chinese', show = True) # Enter any pretrained classification model on Hugging Face
print('sentiment =' , sentiment)
```
#### output
``` python
sentiment = {'none': 2.7154697818332352e-05, 'disgust': 2.6893396352534182e-05, 'happiness': 0.00047770512173883617, 'like': 0.9991452693939209, 'fear': 3.293586996733211e-05, 'sadness': 0.00013537798076868057, 'anger': 8.243478805525228e-05, 'surprise': 7.21854084986262e-05}
```
![Example Image](README_PIC/sentiment.png)

