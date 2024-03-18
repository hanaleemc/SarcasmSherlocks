import pandas as pd
import re
import nltk
import spacy
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')  # needed for POS tagging
nltk.download('wordnet')
nltk.download('stopwords')

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
def preprocess_type_II(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text

    # Convert urls to HTTPURL, user handles to @user token, delete hashtags
    tweet_text = url_regex.sub('HTTPURL', tweet_text)
    tweet_text = mention_regex.sub('@USER', tweet_text)
    tweet_text = hashtag_regex.sub('', tweet_text)

    return tweet_text

"""
******************************************************************************************************
"""

# Preprocessing Type III | Type II + process emoji
def preprocess_type_III(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text

    # Use function to convert urls to HTTPURL, user handles to @user token, delete hashtags
    tweet_text = preprocess_type_II(tweet_text)

    # Replace contractions with full forms
    contraction_mapping = {"isn't": "is not", "’cause": "because", "You'd": "you would", "I’m": "I am",
                           "Couldn't": "Could not"}  # Customize as needed
    for contraction, full_form in contraction_mapping.items():
        tweet_text = tweet_text.replace(contraction, full_form)

    # Convert emotion icons to their string text
    tweet_text = emoji.demojize(tweet_text)

    return tweet_text

"""
******************************************************************************************************
"""

# Preprocessing Type IV_A | Type III + POS tagging, lemmatization, stop words removal
# with SpaCy

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")

def is_special_pattern(token_text):
    """
    Check if the token text is a special pattern we want to keep:
    - Combination of question mark, exclamation mark and dot in any order (e.g., '?!', '!!??', '?!!?')
    """
    if re.fullmatch(r"[\?!]+", token_text):
        return True

    return False


def preprocess_type_IV_A(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text

    tweet_text = preprocess_type_III(tweet_text)

    # Process the tweet text with spaCy
    doc = nlp(tweet_text)

    processed_tokens = [token.text.lower() if is_special_pattern(token.text) else token.lemma_.lower()
                        for token in doc if not token.is_stop and (
                                    is_special_pattern(token.text) or (not token.is_punct and not token.is_space))]

    tweet_text = ' '.join(processed_tokens)

    return tweet_text

"""
******************************************************************************************************
"""
# Preprocessing Type IV_B | Type III + POS tagging, lemmatization, stop words removal
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

def preprocess_type_IV_B(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text

    tweet_text = preprocess_type_III(tweet_text)

    # Tokenize the tweet text
    tokens = word_tokenize(tweet_text)

    # Process tokens
    processed_tokens = [lemmatizer.lemmatize(token).lower() if is_special_pattern(token) else token.lower()
                        for token in tokens if token.lower() not in stop_words and (
                            is_special_pattern(token) or (token.isalpha()))]

    tweet_text = ' '.join(processed_tokens)

    return tweet_text

