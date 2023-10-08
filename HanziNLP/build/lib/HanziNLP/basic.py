import jieba
import re
import logging
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# Suppress informational messages from jieba
logging.getLogger('jieba').setLevel(logging.WARNING)

def count_chars(text):
    """Count the number of Chinese characters in a text."""
    count = len([char for char in text if '\u4e00' <= char <= '\u9fff'])
    return count

def count_words(text):
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
    List the names of all available fonts.
    """
    font_files = os.listdir(FONTS_DIR)
    font_names = [os.path.splitext(font)[0] for font in font_files if font.endswith('.ttf')]
    return font_names

def get_font_path(font_name, show=True):
    """
    Get the file path of the specified font.

    Parameters:
    font_name (str): The name of the font (without file extension).
    show (bool): Whether to display sample text with the font.

    Returns:
    str: The file path of the specified font, or None if the font is not found.
    """
    font_file = f"{font_name}.ttf"
    font_path = os.path.join(FONTS_DIR, font_file)
    if os.path.isfile(font_path):
        if show:
            text = '你好，世界啊'
            prop = fm.FontProperties(fname=font_path, size=40)
            plt.text(0.5, 0.5, text, fontproperties=prop, ha='center', va='center')
            plt.axis('off')
            plt.show()
        return font_path
    else:
        print(f"Font {font_name} not found.")
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
    List available stopword files.

    Returns:
    list: A list of filenames in the stopwords directory.
    """
    return os.listdir(STOPWORDS_DIR)
    
def load_stopwords(file_name):
    """
    Load stopwords from a specified file.
    """
    with open(os.path.join(STOPWORDS_DIR, file_name), 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)

common_stopwords = load_stopwords('common_stopwords.txt')

def word_tokenize(text, stopwords=common_stopwords, text_only=False, include_numbers=True):
    """
    Tokenize Chinese text and remove stopwords.

    Parameters:
    text (str): The input Chinese text.
    stopwords_files (list): A list of filenames containing stopwords. Default is ['common_stopwords.txt'].
    text_only (Boolean): Only tokenize English and Chinese texts if True. Default is False.
    include_numbers (Boolean): Whether to include numbers in the tokenized output. Default is True.

    Returns:
    list: A list of tokens after removing stopwords.
    """
    stopwords_list = set()
    stopwords_list = stopwords_list.union(stopwords)
    
    # Tokenize text
    tokens = jieba.cut(text)

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


