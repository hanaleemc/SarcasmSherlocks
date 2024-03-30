import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
import emoji
nltk.download('wordnet') # for nlp tasks

train_df = pd.read_csv('Train_Dataset.csv')

import numpy as np

#Creating dictionary of words with their corresponding embedding
words = dict() # will be d, words==dictionary of words mapping to their corresponding vector

def add_to_dict(d, filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split(' ')

            try:
                d[line[0]] = np.array(line[1:], dtype=float) # separates word [line[0]]-->key from its embeddings [line[1:]]
            except:
                continue
add_to_dict(words, 'glove.6B.50d.txt') #50 dimensional vectors
#glove can downloaded or unzipped or just upload the text file to git


tokenizer = nltk.RegexpTokenizer(r"\w+") #comes with built in preprocessing steps
#removes special characters


#Preporcess, tokenize and create embeddings
# Load the NLTK tokenizer
tokenizer = nltk.RegexpTokenizer(r"\w+")
lemmatizer = WordNetLemmatizer()

# Define URL, hashtag, and mention regex patterns
url_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
hashtag_regex = re.compile(r'#\S+')
mention_regex = re.compile(r'@\w+')

def preprocess_type_II(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text

    # Convert urls to HTTPURL, user handles to @USER token, delete hashtags
    tweet_text = url_regex.sub('HTTPURL', tweet_text)
    tweet_text = mention_regex.sub('@USER', tweet_text)
    tweet_text = hashtag_regex.sub('', tweet_text)

    return tweet_text

def preprocess_type_III(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text

    # Use function to convert urls to HTTPURL, user handles to @USER token, delete hashtags
    tweet_text = preprocess_type_II(tweet_text)

    # Replace contractions with full forms
    contraction_mapping = {"isn't": "is not", "’cause": "because", "You'd": "you would", "I’m": "I am",
                           "Couldn't": "Could not", }  # Customize as needed
    for contraction, full_form in contraction_mapping.items():
        tweet_text = tweet_text.replace(contraction, full_form)

    # Convert emotion icons to their string text
    tweet_text = emoji.demojize(tweet_text)

    return tweet_text

def message_to_token_list(s, preprocess_func=preprocess_type_III):
    # Apply preprocessing
    s = preprocess_func(s)

    # Tokenize the preprocessed text
    tokens = tokenizer.tokenize(s)
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
    useful_tokens = [t for t in lemmatized_tokens if t in words]

    return useful_tokens


def message_to_word_vectors(message, word_dict=words):
    
    processed_list_of_tokens = message_to_token_list(message) #preprocessing and tokenization step

    vectors = [] # initialize empty array

    for token in processed_list_of_tokens:
        
        if token not in word_dict: #word embedding dictionary
            continue

        token_vector = word_dict[token] #whatever word we saw there grab its corresponding vector
        vectors.append(token_vector)


    return np.array(vectors, dtype=float)

"""
********************************************************************************************************************************************
"""

import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import emoji
import re

# Load NLTK resources
nltk.download('wordnet')
tokenizer = nltk.RegexpTokenizer(r"\w+")
lemmatizer = WordNetLemmatizer()

# Define URL, hashtag, and mention regex patterns
url_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
hashtag_regex = re.compile(r'#\S+')
mention_regex = re.compile(r'@\w+')

# Define words dictionary
words = {}

# Load word embeddings
def load_word_embeddings(filename):
    word_dict = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            try:
                word_dict[line[0]] = np.array(line[1:], dtype=float)
            except:
                continue
    return word_dict

# Preprocess text
def preprocess_text(text):
    if pd.isna(text) or text == '':
        return text
    text = url_regex.sub('HTTPURL', text)
    text = mention_regex.sub('@USER', text)
    text = hashtag_regex.sub('', text)
    return text

#lemmatize tokens
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]

# Function to preprocess and tokenize text
def preprocess_and_tokenize(text, word_embeddings):
    text = preprocess_text(text)
    tokens = tokenizer.tokenize(text)
    tokens = lemmatize_tokens(tokens)
    useful_tokens = [token for token in tokens if token in word_embeddings]
    return useful_tokens

# Function to convert message to word vectors
def message_to_word_vectors(message, word_embeddings):
    tokens = preprocess_and_tokenize(message, word_embeddings)
    vectors = [word_embeddings[token] for token in tokens]
    return np.array(vectors)

# Load word embeddings
words = load_word_embeddings('glove.6B.50d.txt')

# How to implement
train_df = pd.read_csv('Train_Dataset.csv')
message = train_df['tweet'].iloc[0]
word_vectors = message_to_word_vectors(message, words)
