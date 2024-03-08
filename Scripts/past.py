
from past import SoMaJo
paragraphs = ["That aint bad!:D"]
tokenizer = SoMaJo(language="en_PTB")
sentences = tokenizer.tokenize_text(paragraphs)
for sentence in sentences:
    for token in sentence:
        print(token.text)
    print()

#somajo requires preprocessing -> potential way of preprocessing 
    
import unicodedata
import re

def preprocess_text(text):
    # Normalize to Unicode Normalization Form C (NFC)
    normalized_text = unicodedata.normalize('NFC', text)
    
    # Remove control characters and unwanted characters
    cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F\u200B-\u200D\uFEFF]', '', normalized_text)
    
    return cleaned_text

# Example usage:
input_text = "Some text with soft hyphen­ and zero-width space​."
preprocessed_text = preprocess_text(input_text)

print("Original Text:", repr(input_text))
print("Preprocessed Text:", repr(preprocessed_text)) 


#original code 
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import bert
import pandas as pd 
import random 


#train_path = '../../Data/Train_Dataset.csv'
DATA_PATH = './Data/train.En.csv'
test_path = '../../Data/Test_Dataset.csv'


#data processing functions 

def dataset_embedding(dataset_path, tokenizer, batch_size=32):
    dataset = pd.read_csv(dataset_path)[["tweet", "sarcastic"]]
    dataset = dataset[dataset['tweet'].notna()]
    
    tokenized_tweets = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweet)) for tweet in dataset['tweet']]
    
    tweets_with_len = [[tweet, dataset['sarcastic'].iloc[i], len(tweet)] for i, tweet in enumerate(tokenized_tweets)]
    random.Random(42).shuffle(tweets_with_len)
    
    tweets_with_len.sort(key=lambda x: x[2])
    sorted_tweets_labels = [(tweet_lab[0], tweet_lab[1]) for tweet_lab in tweets_with_len] # remove tweet len
    processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_tweets_labels, output_types=(tf.int32, tf.int32))
    
    return processed_dataset.padded_batch(batch_size, padded_shapes=((None, ), ()))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#class weights enables setting weights for imbalanced classes 

#dataset tokensizer this code prepares datasets for training and testing by initializing 
#a BERT tokenizer using a pre-trained BERT model from TensorFlow Hub and then using this 
#tokenizer to tokenize and embed the text data in the training and testing datasets. 

def prepare_datasets(train_path, test_path):
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    
    dataset_train = dataset_embedding(train_path, tokenizer)
    dataset_test = dataset_embedding(test_path, tokenizer)
    
    return dataset_train, dataset_test, tokenizer




#load needed packages 
import numpy as np
import pandas as pd
import csv
import nltk
import emoji
import re

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

#Step 0 | Load data [Pandas]
DATA_PATH = './Data/train.En.csv'
df = pd.read_csv(DATA_PATH, usecols= ['tweet','sarcastic'])



#Group 1 | Remove [URLs, Hastags, Stop words] & Keep  [Punctuation; Userhandles; emojis]
 
def plumeria_type_II(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text
    
    # Convert emotion icons to their string text
    tweet_text = emoji.demojize(tweet_text)
    
    # Convert URLs to "HTTPURL" token
    tweet_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'HTTPURL', tweet_text)

    # Convert mentions to "@USER" token
    tweet_text = re.sub(r'@\w+', '@USER', tweet_text)

    return tweet_text


#need to add in tokenizing step 

'''def tokenize_and_lemmatize(data):
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

    return processed_data'''





# Apply the preprocessing function to the 'Tweet' column
df['ProcessedTweet_II'] = df['tweet'].apply(preprocess_type_II)

#Create new dataframe with processed tweets 
df_II = df[['ProcessedTweet_II', 'sarcastic']]
print(df_II)

def preprocess_type_III(tweet_text):
    smiley_mapping = {":-)": "smiley", ":)": "smiley", ":-(": "sad", ":(": "sad", ":D": "playful"}
    if pd.notna(tweet_text):
        for smiley_code, value in smiley_mapping.items():
            tweet_text = tweet_text.replace(smiley_code, value)

        # Remove more than two successive occurrences of any punctuation
        tweet_text = re.sub(r'[^\w\s]{2,}', ' ', tweet_text)

        # Remove more than two successive occurrences of the same character
        tweet_text = re.sub(r'(\w)\1{2,}', r'\1\1', tweet_text)

        # Replace contractions with full forms
        contraction_mapping = {"isn't": "is not", "’cause": "because","You'd": "you would", "I’m": "I am", "Couldn't": "Could not"}  # Customize as needed
        for contraction, full_form in contraction_mapping.items():
            tweet_text = tweet_text.replace(contraction, full_form)

        return tweet_text

# Apply the preprocessing function to the 'tweet' column
df['ProcessedTweet_III'] = df['tweet'].apply(preprocess_type_III)

# Create a new dataframe with processed tweets and 'sarcastic' column
df_III = df[['ProcessedTweet_III', 'sarcastic']]

# Display the result
print(df_III)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def preprocess_type_IV(tweet_text, language='english'):
    if pd.notna(tweet_text):
        lemmatizer = WordNetLemmatizer()
        tweet_text = ' '.join([lemmatizer.lemmatize(word) for word in tweet_text.split()])

        # Remove stopwords (for English)
        stop_words = set(stopwords.words(language))
        tweet_text = ' '.join([word for word in tweet_text.split() if word.lower() not in stop_words])

        return tweet_text


# Apply the preprocessing function to the 'Tweet' column
df['ProcessedTweet_IV'] = df['tweet'].apply(preprocess_type_IV)

#Create new dataframe with processed tweets 
df_IV = df[['ProcessedTweet_IV', 'sarcastic']]

print(df_IV)


import Tests.preprocessing as preprocessing

import tensorflow as tf
from transformers import AutoTokenizer
import pandas as pd 
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
import re
import nltk
from nltk import TweetTokenizer
import spacy
import emoji

def preprocess_type_I(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text
    # Convert emotion icons to their string text
    tweet_text = emoji.demojize(tweet_text)

    # Convert URLs to "HTTPURL" token
    tweet_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'HTTPURL', tweet_text)

    # Convert mentions to "@USER" token
    tweet_text = re.sub(r'@\w+', '@USER', tweet_text)

    tknzr = TweetTokenizer()
    tokenized_tweet = tknzr.tokenize(tweet_text)

    return tokenized_tweet

train_data = './Data/train.En.csv'
test_data = './Data/test.En.csv'
train_df = pd.read_csv(train_data, usecols=['tweet', 'sarcastic'])
#test_df = pd.read_csv(test_data, usecols=['tweet', 'sarcastic'])

train_df['ProcessedTweet'] = train_df['tweet'].apply(preprocess_type_I)
train_df = train_df[['ProcessedTweet','sarcastic']]
#train_data = train_df['ProcessedTweet'].to_numpy()
train_data = train_df['ProcessedTweet']
test = dict(train_data)
print(test)

train_label = train_df['sarcastic']

train_data = tf.data.Dataset.from_tensor_slices((train_data))
train_label = tf.data.Dataset.from_tensor_slices((train_label))

lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(100, 32),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

lstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#lstm.compile(loss='SigmoidFocalCrossEntropy',optimizer='adam',metrics=['accuracy'])

print(lstm.summary())

lstm.fit(train_data, epochs=10, validation_data=train_label, class_weight={1:4, 0:1})

