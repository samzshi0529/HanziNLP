# HanziNLP

HanziNLP is a Natural Language Processing (NLP) package designed specifically for Chinese language processing. It offers a range of features, from tokenization and sentiment analysis to visualization tools, making it a comprehensive solution for Chinese text analysis.

## Installation

Install HanziNLP using pip:

```shell
pip install HanziNLP
```

## Features

- **Chinese Tokenization**: Efficiently tokenize Chinese text into words or phrases.
- **Sentiment Analysis**: Analyze the sentiment of Chinese text with built-in sentiment models.
- **Stopword Removal**: Remove unnecessary stopwords from the text to enhance text analysis.
- **Chinese Font Handling**: Visualize Chinese text using various fonts seamlessly.
- **Sentence Segmentation**: Break down large text into manageable sentences for detailed analysis.
- ... _and more!_

## Usage
### Tokenization
```shell
from HanziNLP import word_tokenize

text = "我今天很开心"
tokens = word_tokenize(text)
print(tokens)  # Output: ['我', '今天', '很', '开心']
```

## License
HanziNLP is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
