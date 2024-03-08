import pandas as pd
import re
import nltk
from nltk import TweetTokenizer
import spacy
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')  # needed for POS tagging
nltk.download('wordnet')
nltk.download('stopwords')

# Step 0 | Load data [Pandas]
DATA_PATH = './train.En.csv'
df = pd.read_csv(DATA_PATH, usecols=['tweet', 'sarcastic'])

# Preprocessing Type I | Convert emotion icons, urls & convert to @user token (Shaheen & Nigam, SemEval 2022)
def preprocess_type_I(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text
    # Convert emotion icons to their string text
    tweet_text = emoji.demojize(tweet_text)

    # Convert URLs to "HTTPURL" token
    tweet_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'HTTPURL',
                        tweet_text)

    # Convert mentions to "@USER" token
    tweet_text = re.sub(r'@\w+', '@USER', tweet_text)

    return tweet_text

"""
******************************************************************************************************
"""
url_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
hashtag_regex = re.compile(r'#\S+')
mention_regex = re.compile(r'@\w+')

# Preprocessing Type II | Convert urls to HTTPURL, user handles to @user token, delete hashtags
def text_low_processing(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text

    # Convert urls to HTTPURL, user handles to @user token, delete hashtags
    tweet_text = url_regex.sub('HTTPURL', tweet_text)
    tweet_text = mention_regex.sub('@USER', tweet_text)
    tweet_text = hashtag_regex.sub('', tweet_text)

    return tweet_text

def preprocess_type_II(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text

    tweet_text = text_low_processing(tweet_text)

    tknzr = TweetTokenizer()
    tokenized_tweet = tknzr.tokenize(tweet_text)

    return tokenized_tweet

"""
******************************************************************************************************
"""

# Preprocessing Type III | Type II + process emoji
def medium_level_preprocessing(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text

    # Use function to convert urls to HTTPURL, user handles to @user token, delete hashtags
    tweet_text = text_low_processing(tweet_text)

    # Replace contractions with full forms
    contraction_mapping = {"isn't": "is not", "’cause": "because", "You'd": "you would", "I’m": "I am",
                           "Couldn't": "Could not"}  # Customize as needed
    for contraction, full_form in contraction_mapping.items():
        tweet_text = tweet_text.replace(contraction, full_form)

    # Convert emotion icons to their string text
    tweet_text = emoji.demojize(tweet_text)

    return tweet_text

def preprocess_type_III(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text

    tweet_text = medium_level_preprocessing(tweet_text)

    tknzr = TweetTokenizer()
    tokenized_tweet = tknzr.tokenize(tweet_text)

    return tokenized_tweet

"""
******************************************************************************************************
"""
# Preprocessing Type IV | Type III + POS tagging, lemmatization, stop words removal
# with SpaCy

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")


def is_special_pattern(token_text):
    """
    Check if the token text is a special pattern we want to keep:
    - Combination of question mark, exclamation mark and dot in any order (e.g., '?!', '!!??', '?!!?')
    """
    if re.fullmatch(r"[\?!\.]+", token_text):
        return True

    return False


def preprocess_type_IV(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text

    tweet_text = medium_level_preprocessing(tweet_text)

    # Process the tweet text with spaCy
    doc = nlp(tweet_text)

    processed_tokens = [token.text.lower() if is_special_pattern(token.text) else token.lemma_.lower()
                        for token in doc if not token.is_stop and (
                                    is_special_pattern(token.text) or (not token.is_punct and not token.is_space))]

    return processed_tokens


# Preprocessing Type IV | Type III + POS tagging, lemmatization, stop words removal
# with NLTK

# Initialize the NLTK lemmatizer and stop words list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def is_special_pattern(token_text):
    """
    Check if the token text is a special pattern we want to keep:
    - Combination of question mark, exclamation mark and dot in any order (e.g., '?!', '!!??', '?!!?')
    """
    if re.fullmatch(r"[\?!\.]+", token_text):
        return True
    return False

def preprocess_type_IV(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text

    tweet_text = medium_level_preprocessing(tweet_text)

    # Tokenize the tweet text
    tokens = word_tokenize(tweet_text)

    # Process tokens
    processed_tokens = [lemmatizer.lemmatize(token).lower() if is_special_pattern(token) else token.lower()
                        for token in tokens if token.lower() not in stop_words and (
                            is_special_pattern(token) or (token.isalpha()))]

    return processed_tokens
