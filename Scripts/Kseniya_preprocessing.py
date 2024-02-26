import csv
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import emoji
from gensim.models import Word2Vec

TRAIN_PATH = './train.En.csv'

def read_csv_file(filepath):
    data = []
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

tweets_raw = read_csv_file(TRAIN_PATH)


def delete_urls_hashtags_userhandles(data):
    """
    Delete URLs, hashtags and user handles from each cell in every row and column of the data.

    :param data: The data read from the CSV file.
    :return: The data with URLs, hashtags and user handles removed from every cell.
    """

    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    hashtag_pattern = re.compile(r'#\S+')
    user_handle_pattern = re.compile(r'@\w+')

    for row in data:
        for i, cell in enumerate(row):
            cell = url_pattern.sub('', cell)  # Remove URLs
            cell = hashtag_pattern.sub('', cell)  # Remove hashtags
            cell = user_handle_pattern.sub('USER_MENTION', cell)
            row[i] = cell

    return data


def tokenize_and_lemmatize(data):
    """
    Tokenize and lemmatize text data from the second and fourth columns of a data read from a CSV file,
    maintaining the structure of the original CSV file with 5 columns.

    :param data: The data read from the CSV file.
    :return: A list of rows, each preserving the original structure with lemmatized and tokenized
    text in the second and fourth columns.
    """
    data = delete_urls_hashtags_userhandles(data)

    lemmatizer = WordNetLemmatizer()
    processed_data = []

    for row in data:
        # Tokenize and lemmatize the text in the second and fourth columns
        row[1] = [lemmatizer.lemmatize(token) for token in word_tokenize(row[1])]
        row[3] = [lemmatizer.lemmatize(token) for token in word_tokenize(row[3])]
        processed_data.append(row)

    return processed_data


def remove_stop_words(data):
    """
    Remove stop words from the tokenized and lemmatized text data in the second and fourth columns.

    :param data: The data read from the CSV file.
    :return: Processed data with stop words removed from the specified columns.
    """
    data = tokenize_and_lemmatize(data)

    # Load English stop words
    stop_words = set(stopwords.words('english'))

    for row in data:
        # Filter out stop words in the second and fourth columns (indexes 1 and 3)
        row[1] = [token for token in row[1] if token.lower() not in stop_words]
        row[3] = [token for token in row[3] if token.lower() not in stop_words]

    return data

def replace_emojis_in_tokens(processed_data):
    """
    Replace emojis in a list of tokens with their descriptions.

    :param tokens: A list of tokens, potentially containing emojis.
    :return: A list of tokens where emojis have been replaced by their descriptions.
    """

    data = remove_stop_words(processed_data)

    for row in data:
        # Convert each token by replacing emojis with their descriptions
        row[1] = [emoji.demojize(token, delimiters=("", "")) for token in row[1]]
        row[3] = [emoji.demojize(token, delimiters=("", "")) for token in row[3]]

    return processed_data

def train_word2vec_model(processed_data):
    """
    Train a Word2Vec model using the preprocessed text data.

    :param processed_data: The data returned by the replace_emojis_in_tokens function.
    :return: A trained Word2Vec model.
    """
    # Extract text data for Word2Vec training
    sentences = [row[1] for row in processed_data] + [row[3] for row in processed_data]

    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    return model

# Load data and preprocess it (all functions)
replaced_emoji_data = replace_emojis_in_tokens(tweets_raw)

# Train the Word2Vec model
word2vec_model = train_word2vec_model(replaced_emoji_data)

