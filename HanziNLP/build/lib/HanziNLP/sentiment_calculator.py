import jieba
import re
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Suppress informational messages from jieba
logging.getLogger('jieba').setLevel(logging.WARNING)

# Define the path to the fonts directory
FONTS_DIR = os.path.join(os.path.dirname(__file__), 'fonts')

def list_fonts():
    """
    List the names of all available fonts.
    """
    font_files = os.listdir(FONTS_DIR)
    font_names = [os.path.splitext(font)[0] for font in font_files if font.endswith('.otf')]
    return font_names

def get_font_path(font_name):
    """
    Get the file path of the specified font.

    Parameters:
    font_name (str): The name of the font (without file extension).

    Returns:
    str: The file path of the specified font, or None if the font is not found.
    """
    font_file = f"{font_name}.otf"
    font_path = os.path.join(FONTS_DIR, font_file)
    if os.path.isfile(font_path):
        return font_path
    else:
        print(f"Font {font_name} not found.")
        return None
        
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
    return word_count, segmented_text



