# Import necessary packages
import jieba
import re
import logging
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import os
from collections import Counter
from collections import defaultdict
import ipywidgets as widgets
import pandas as pd
import numpy as np
from IPython.display import display, clear_output
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import fasttext
import fasttext.util
from transformers import AutoTokenizer, BertModel, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import seaborn as sns

# Suppress informational messages from jieba
logging.getLogger('jieba').setLevel(logging.WARNING)

def char_freq(text, text_only=True):
    """Count the number of characters in a text."""
    chinese_chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
    chinese_count = len(chinese_chars)
    
    # Remove Chinese characters from text to leave behind English words and others
    text_without_chinese = re.sub("[\u4e00-\u9fff]+", " ", text)
    
    if text_only:
        # Count only alphanumeric English words
        english_words = re.findall(r'\b\w+\b', text_without_chinese)
        english_count = len(english_words)
    else:
        # Count all characters excluding Chinese characters
        english_count = len(text_without_chinese)
    
    total_count = chinese_count + english_count
    return total_count

def word_freq(text):
    """
    Count the number of Chinese and English words in a text using word segmentation,
    and return segmented words.
    """
    words = list(jieba.cut(text))
    
    chinese_words = [word for word in words if re.match("[\u4e00-\u9fff]+", word)]
    
    # Remove Chinese characters from text to leave behind English words and others
    text_without_chinese = re.sub("[\u4e00-\u9fff]+", " ", text)
    english_words = re.findall(r'\b\w+\b', text_without_chinese)
    
    # Combine segmented Chinese words with English words
    total_words = chinese_words + english_words
    word_count = len(total_words)
    segmented_text = ' '.join(total_words)
    return word_count

# Define the path to the fonts directory
FONTS_DIR = os.path.join(os.path.dirname(__file__), 'fonts')

def list_fonts():
    """
    List the names of all available fonts in an interactive table with a button to confirm the selection.

    Returns:
    None
    """
    font_files = os.listdir(FONTS_DIR)
    font_names = [os.path.splitext(font)[0] for font in font_files if font.endswith(('.ttf', '.otf'))]
    df = pd.DataFrame(sorted(font_names), columns=['Font Names'])

    text = widgets.Text(
        value='',
        placeholder='Type something',
        description='Filter:',
        disabled=False
    )

    button = widgets.Button(description="Confirm")
    output = widgets.Output()

    # Function to handle button click and filter table
    def on_button_click(b):
        with output:
            clear_output(wait=True)
            filter_text = text.value
            if filter_text:
                display_df = df[df['Font Names'].str.contains(filter_text, case=False, na=False)]
                print(display_df)
            else:
                print(df)

    button.on_click(on_button_click)

    # Creating a horizontal box with the text and button widgets
    hbox = widgets.HBox([text, button])

    # Initial display of widgets and DataFrame
    display(hbox, output)
    with output:
        print(df)  # Initial DataFrame output


def get_font(font_name, show=True):
    """
    Get the file path of the specified font.

    Parameters:
    font_name (str): The name of the font (without file extension).
    show (bool): Whether to display sample text with the font.

    Returns:
    str: The file path of the specified font, or None if the font is not found.
    """
    ttf_font_file = f"{font_name}.ttf"
    otf_font_file = f"{font_name}.otf"
    ttf_font_path = os.path.join(FONTS_DIR, ttf_font_file)
    otf_font_path = os.path.join(FONTS_DIR, otf_font_file)

    font_path = None
    if os.path.isfile(ttf_font_path):
        font_path = ttf_font_path
    elif os.path.isfile(otf_font_path):
        font_path = otf_font_path

    if font_path:
        if show:
            text = '你好，世界'
            prop = fm.FontProperties(fname=font_path, size=40)
            plt.text(0.5, 0.5, text, fontproperties=prop, ha='center', va='center')
            plt.axis('off')
            plt.show()
        return font_path
    else:
        print(f"Font {font_name} not found in both .ttf and .otf formats.")
        return None

# Define the path to the stopwords directory
STOPWORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stopwords')

def sentence_segment(text):
    """
    Segment a large text into sentences.

    Parameters:
    text (str): The input text.

    Returns:
    list: A list of sentence strings.
    """
    # Adding a space after each punctuation to handle cases with no spacing
    text = re.sub(r'([。.!?！？])', r'\1 ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Define a regular expression pattern to split the text at sentence-ending punctuation marks
    pattern = re.compile(r'(?<=[。.!?！？])\s')
    # Split the text using the pattern
    sentences = re.split(pattern, text)
    # Remove any trailing spaces in each sentence
    sentences = [sentence.strip() for sentence in sentences if sentence]

    # Return the list of sentences
    return sentences

def list_stopwords():
    """
    List available stopword files with an interactive search feature.

    Returns:
    None
    """
    stopwords_files = os.listdir(STOPWORDS_DIR)
    df = pd.DataFrame(sorted(stopwords_files), columns=['Stopword Files'])

    text = widgets.Text(
        value='',
        placeholder='Type something',
        description='Filter:',
        disabled=False
    )

    button = widgets.Button(description="Confirm")
    output = widgets.Output()

    # Function to handle button click and filter table
    def on_button_click(b):
        with output:
            clear_output(wait=True)
            filter_text = text.value
            if filter_text:
                display_df = df[df['Stopword Files'].str.contains(filter_text, case=False, na=False)]
                print(display_df)
            else:
                print(df)

    button.on_click(on_button_click)

    # Creating a horizontal box with the text and button widgets
    hbox = widgets.HBox([text, button])

    # Initial display of widgets and DataFrame
    display(hbox, output)
    with output:
        print(df)  # Initial DataFrame output
    
def load_stopwords(file_name):
    """
    Load stopwords from a specified file.
    """
    with open(os.path.join(STOPWORDS_DIR, file_name), 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)

common_stopwords = load_stopwords('common_stopwords.txt')

def word_tokenize(text, mode='precise', stopwords=common_stopwords, text_only=False, include_numbers=True):
    """
    Tokenize Chinese text and remove stopwords.

    Parameters:
    text (str): The input Chinese text.
    mode (str): Tokenization mode ('all', 'precise', or 'search_engine'). Default is 'precise'.
    stopwords_files (list): A list of filenames containing stopwords. Default is ['common_stopwords.txt'].
    text_only (Boolean): Only tokenize English and Chinese texts if True. Default is False.
    include_numbers (Boolean): Whether to include numbers in the tokenized output. Default is True.

    Returns:
    list: A list of tokens after removing stopwords.
    """
    stopwords_list = set()
    stopwords_list = stopwords_list.union(stopwords)
    
    # Validate the mode parameter
    valid_modes = ['all', 'precise', 'search_engine']
    if mode not in valid_modes:
        raise ValueError("Invalid mode. Choose from 'all', 'precise', or 'search_engine'.")

    # Choose the tokenization function based on the mode
    if mode == 'all':
        tokens = jieba.cut(text, cut_all=True)
    elif mode == 'precise':
        tokens = jieba.cut(text, cut_all=False)
    elif mode == 'search_engine':
        tokens = jieba.cut_for_search(text)

    if text_only:
        # If text_only is True, retain only Chinese and English characters in tokens
        # This regex pattern matches Chinese characters, English words, and optionally, numbers based on include_numbers
        pattern_str = r'[\u4e00-\u9fff\w]+' if include_numbers else r'[\u4e00-\u9fffA-Za-z_]+'
        pattern = re.compile(pattern_str)

        # Use the regex pattern to filter the tokens
        tokens = [token for token in tokens if pattern.fullmatch(token)]

    # Remove stopwords
    processed_tokens = [token for token in tokens if token not in stopwords_list]
    
    return processed_tokens

def BoW(segmented_text_list):
    """
    Convert a list of segmented texts into a Bag of Words (BoW) representation.

    Parameters:
    segmented_text_list (list of str): A list of segmented texts.

    Returns:
    dict: A dictionary representing word frequencies.
    """
    word_frequencies = Counter()

    for text in segmented_text_list:
        # Using the word_freq function to get the count of each word in the text
        # You might want to modify the word_freq function to return not just the count but also the words
        words = list(jieba.cut(text))  # Assuming you've segmented the text using jieba.cut
        word_frequencies.update(words)
    
    return dict(word_frequencies)
    

def ngrams(tokens, n=3):
    """
    Convert a list of tokens into n-grams and count their frequencies.

    Parameters:
    tokens (list): The input list of tokens.
    n (int): The number for n-grams. Default is 2 (bigrams).

    Returns:
    dict: A dictionary with n-grams as keys and their frequencies as values.
    """
    if n <= 0:
        raise ValueError("n should be greater than 0")

    # Check if tokens is a list
    if not isinstance(tokens, list):
        raise TypeError("Input should be a list of tokens")

    # Create n-grams
    ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    # Count frequencies of each n-gram
    freq_dist = defaultdict(int)
    for ngram in ngrams:
        freq_dist[ngram] += 1
    
    return dict(freq_dist)

def TF_IDF(text_list, max_features=None):
    """
    Transform a list of texts into their TF-IDF representation using scikit-learn's TfidfVectorizer.

    Parameters:
    text_list (list of str): A list of texts to be transformed.
    max_features (int, optional): Maximum number of features (terms) to be extracted. Defaults to None (all features).

    Returns:
    sparse_matrix: A sparse matrix of shape (n_samples, n_features).
    feature_names: List of feature names.
    """

    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, max_features=max_features)

    # Fit and transform the text_list
    tfidf_matrix = vectorizer.fit_transform(text_list)

    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix, feature_names

def TT_matrix(tokenized_texts, window_size=1):
    """
    Generates a term-term matrix from a list of tokenized texts.

    Parameters:
    tokenized_texts (list of list of str): A list of tokenized texts.
    window_size (int): The window size for co-occurrence. Default is 1.

    Returns:
    np.array: A square matrix where entry (i, j) is the co-occurrence between term i and term j.
    index_to_term: A dictionary mapping from index to term.
    """

    co_occurrences = Counter()
    term_to_index = {}
    index_to_term = {}

    # Count co-occurrences
    for tokens in tokenized_texts:
        for i, term1 in enumerate(tokens):
            for j in range(max(i-window_size, 0), min(i+window_size+1, len(tokens))):
                if i != j:
                    term2 = tokens[j]
                    pair = tuple(sorted([term1, term2]))
                    co_occurrences[pair] += 1

    # Create term to index mapping
    for term1, term2 in co_occurrences.keys():
        if term1 not in term_to_index:
            idx = len(term_to_index)
            term_to_index[term1] = idx
            index_to_term[idx] = term1
        if term2 not in term_to_index:
            idx = len(term_to_index)
            term_to_index[term2] = idx
            index_to_term[idx] = term2

    # Create and fill the co-occurrence matrix
    matrix_size = len(term_to_index)
    matrix = np.zeros((matrix_size, matrix_size))
    for (term1, term2), count in co_occurrences.items():
        idx1 = term_to_index[term1]
        idx2 = term_to_index[term2]
        matrix[idx1, idx2] = count
        matrix[idx2, idx1] = count  # Symmetric matrix

    return matrix, index_to_term

def text_similarity(text1, text2, method='cosine'):
    """
    Computes similarity between two texts using specified method.

    Parameters:
    text1 (str): The first text.
    text2 (str): The second text.
    method (str): The method to use for computing similarity ('cosine', 'jaccard', 'euclidean', or 'levenshtein'). Default is 'cosine'.

    Returns:
    float: The similarity score between the two texts.
    """

    # Tokenize texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)

    # Vectorize texts
    vector1 = Counter(tokens1)
    vector2 = Counter(tokens2)

    # Convert counters to vectors
    all_tokens = set(vector1.keys()).union(set(vector2.keys()))
    vec1 = np.array([vector1[token] for token in all_tokens])
    vec2 = np.array([vector2[token] for token in all_tokens])

    # Compute similarity
    if method == 'cosine':
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm1 * norm2)
    elif method == 'jaccard':
        intersection = len(set(tokens1) & set(tokens2))
        union = len(set(tokens1) | set(tokens2))
        similarity = intersection / union
    elif method == 'euclidean':
        distance = np.linalg.norm(vec1 - vec2)
        similarity = 1 / (1 + distance)
    elif method == 'levenshtein':
        if len(text1) > len(text2):
            text1, text2 = text2, text1
        distances = range(len(text1) + 1)
        for index2, char2 in enumerate(text2):
            new_distances = [index2+1]
            for index1, char1 in enumerate(text1):
                if char1 == char2:
                    new_distances.append(distances[index1])
                else:
                    new_distances.append(1 + min((distances[index1], distances[index1+1], new_distances[-1])))
            distances = new_distances
        similarity = 1 / (1 + distances[-1])
    else:
        raise ValueError("Invalid method. Choose 'cosine', 'jaccard', 'euclidean', or 'levenshtein'.")

    return similarity

def Word2Vec(text, dimension=300):
    """
    Get embeddings for text containing both English and Chinese words.

    Parameters:
    text (str): The input text which may contain both English and Chinese words.
    dimension (int): Dimensionality of the resulting word embeddings. Default is 300.

    Returns:
    list of numpy.ndarray: A list of word embeddings for each word in the text.
    """
    # Download the English and Chinese models if they are not downloaded
    fasttext.util.download_model('en', if_exists='ignore')
    fasttext.util.download_model('zh', if_exists='ignore')

    # Load the models
    en_model = fasttext.load_model('cc.en.300.bin')
    zh_model = fasttext.load_model('cc.zh.300.bin')

    # Reduce model dimensionality if requested
    if dimension < 300:
        fasttext.util.reduce_model(en_model, dimension)
        fasttext.util.reduce_model(zh_model, dimension)

    # Tokenize and get embeddings
    embeddings = []
    words = word_tokenize(text)  # tokenizer

    for word in words:
        if re.match("[\u4e00-\u9fff]+", word):  # if the word contains Chinese characters
            embeddings.append(zh_model.get_word_vector(word))
        else:
            embeddings.append(en_model.get_word_vector(word))
    
    # Unload models to free up memory
    del en_model
    del zh_model
    
    return embeddings

def get_bert_embeddings(text, model="bert-base-chinese"):
    """
    Get BERT embeddings for the specified text using a pre-trained Chinese BERT model.

    Parameters:
    text (str): The input text.
    model_name (str): The name of the pre-trained Chinese BERT model to use. Default is "bert-base-chinese."

    Returns:
    sentence_embedding (list): The sentence embedding as a list of floats.
    tokens (list): The tokens associated with the sentence embedding.
    """
    # Load the pre-trained Chinese BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = BertModel.from_pretrained(model)

    # Tokenize the text and prepare it for the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Make predictions using the model
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    sentence_embedding = torch.mean(last_hidden_states, dim=1).numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return sentence_embedding, tokens

def lda_model(texts, num_topics=10, passes=15, dictionary=None):
    """
    Train an LDA model on the provided texts to extract topics.

    Parameters:
    texts (list of list of str): A list of documents, each document represented as a list of tokens.
    num_topics (int): Number of topics to extract.
    passes (int): Number of training passes.
    dictionary (corpora.Dictionary, optional): Precomputed Gensim dictionary.

    Returns:
    lda_model: Trained LDA model.
    corpus: Corpus used to train the model.
    dictionary: Dictionary used to train the model.
    """
    # If no dictionary is provided, create a new one from the provided texts
    if dictionary is None:
        dictionary = corpora.Dictionary(texts)
    
    # Convert texts to a bag-of-words representation
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Train the LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )
    
    return lda_model, corpus, dictionary

def print_topics(lda_model, num_words=10):
    """
    Print the top words associated with each topic from the trained LDA model.

    Parameters:
    lda_model: Trained LDA model.
    num_words (int): Number of top words to display for each topic.

    Returns:
    None
    """
    for idx, topic in lda_model.print_topics(-1, num_words):
        print(f"Topic: {idx} \nWords: {topic}")

def sentiment(text, model='hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2', print_all=True, show=False):
    """
    Perform sentiment analysis on the input text using the specified model.

    Parameters:
    text (str): The input text to be analyzed.
    model (str): The name of the pre-trained model to use.
    print_all (bool): Whether to print probabilities for all labels or only the label with the highest probability.
    show (bool): Whether to plot a bar chart showing the probability distribution for each label.

    Returns:
    dict or tuple: A dictionary containing the sentiment labels and their corresponding probabilities if print_all is True,
                   otherwise a tuple containing the label with the highest probability and its corresponding probability.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model)

    # Tokenize input text and obtain model output
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    # Get probabilities using softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get the list of labels from the model’s configuration
    labels = model.config.id2label.values()

    # Create a dictionary to store the probabilities associated with each label
    sentiment_probs = {label: prob.item() for label, prob in zip(labels, probs[0])}

    if show:
        # Set the aesthetic style of the plots
        sns.set_style("whitegrid")

        # Set the color palette
        sns.set_palette("husl")

        # Plot a horizontal bar chart
        plt.figure(figsize=(5, 5))
        ax = sns.barplot(x=list(sentiment_probs.values()), y=list(sentiment_probs.keys()))
        plt.xlabel('Probability', fontsize=7)
        plt.ylabel('Sentiment', fontsize=7)
        plt.title('Sentiment Probability Distribution', fontsize=9)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        # Hide the right, top, and bottom axes
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Remove x-axis grid lines
        ax.xaxis.grid(False)

        # Add data values on each bar
        for index, value in enumerate(sentiment_probs.values()):
            ax.text(x=value, y=index, s=f"{value:.2f}", va='center', fontsize=7)

        plt.show()

    if print_all:
        return sentiment_probs
    else:
        # Find the label with the highest probability
        max_label = max(sentiment_probs, key=sentiment_probs.get)
        return max_label, sentiment_probs[max_label]

def dashboard():
    # Title
    title = widgets.HTML(
        value="<h1>HanziNLP Dashboard</h1>",
        placeholder='',
        description='',
    )

    # Label for text input
    text_label = widgets.HTML(
        value="<b style='font-size: 15px;'>Text Input:</b>",
        placeholder='',
        description='',
    )

    # Text input widget
    text_input = widgets.Textarea(
        value='',
        placeholder='Enter Text Here',
        disabled=False,
        layout=widgets.Layout(height='30px', width='auto')
    )

    # Refresh button
    button = widgets.Button(
        description="Confirm",
        layout=widgets.Layout(height='30px', width='auto')
    )

    # Text display area
    text_display = widgets.Textarea(
        value='This is the text you inputted',
        disabled=True,
        layout=widgets.Layout(height='400px', width='auto')
    )

    # Table to display counts
    count_trace = go.Table(
        header=dict(values=['Metric', 'Count']),
        cells=dict(values=[['Character Count', 'Word Count'], [0, 0]])
    )
    count_table = go.FigureWidget([count_trace], layout={'height': 300, 'width': '500px'})

    # Table to display top word frequencies
    freq_trace = go.Table(
        header=dict(values=['Token', 'Frequency']),
        cells=dict(values=[[], []])
    )
    freq_table = go.FigureWidget([freq_trace], layout={'height': 300, 'width': '500px'})

    # Update function
    def update_table(button):
        text = text_input.value
        char_count_value = char_freq(text)
        word_count_value = word_freq(text)
        count_table.data[0].cells.values = [['Character Count', 'Word Count'], [char_count_value, word_count_value]]
        
        # Tokenize and get frequencies
        tokens = word_tokenize(text)
        token_freq = Counter(tokens)
        top_tokens = token_freq.most_common(20)  # Get top 10 tokens
        tokens, frequencies = zip(*top_tokens)
        freq_table.data[0].cells.values = [tokens, frequencies]
        
        # Update text display area
        text_display.value = f"This is the text you inputted:\n{text}"

    # Set button click event handler
    button.on_click(update_table)

    # Layout widgets
    input_and_button = widgets.HBox([text_input, button], layout=widgets.Layout(align_items='center'))
    tables = widgets.HBox([count_table, freq_table])
    user_interface = widgets.VBox([input_and_button, text_display, tables])
    
    display(widgets.VBox([title, user_interface]))

