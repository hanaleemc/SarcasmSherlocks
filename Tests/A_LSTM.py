
### LSTM & UTILS from ABAKOSHI | Slightly altered 


import tensorflow as tf
import random
import pandas as pd
from keras import backend as K
from transformers import DistilBertTokenizer, TFDistilBertModel

#function from utils 

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

def dataset_embedding(dataset_path, tokenizer, batch_size=32):
    dataset = pd.read_csv(dataset_path)[["tweet", "sarcastic"]]
    dataset = dataset[dataset['tweet'].notna()]
    
    '''
    The following lines create word embeddings for each individual tweet, 
    then process the dataset into tf.data.Dataset.from_generator with the labels & tokeized tweets 
    this puts it into a format which will work with the lstm model (with the added padded_batch step).
    This is the part I dont really understand and I think we'll need to figure out so that we can 
    implement our preprocessing steps/logic -> get it into the correct shape for the lstm model

    '''
    tokenized_tweets = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweet)) for tweet in dataset['tweet']]
    tweets_with_len = [[tweet, dataset['sarcastic'].iloc[i], len(tweet)] for i, tweet in enumerate(tokenized_tweets)]
    random.Random(42).shuffle(tweets_with_len)
    tweets_with_len.sort(key=lambda x: x[2])
    sorted_tweets_labels = [(tweet_lab[0], tweet_lab[1]) for tweet_lab in tweets_with_len] # remove tweet len

    processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_tweets_labels, output_types=(tf.int32, tf.int32))
    
    return processed_dataset.padded_batch(batch_size, padded_shapes=((None, ), ()))

def prepare_datasets(train_path, test_path):

    #replaced their tokenizer with distil bert-base-uncased 
    model_name = 'distilbert-base-uncased'

    # Initialize the tokenizer and model specifically for DistilBERT
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    #model = TFDistilBertModel.from_pretrained(model_name)
    
    dataset_train = dataset_embedding(train_path, tokenizer)
    dataset_test = dataset_embedding(test_path, tokenizer)
    
    return dataset_train, dataset_test, tokenizer

# LSTM Model  

train_path = './Data/train.En.csv'
test_path = './Data/test.En.csv'

train_data, test_data, tokenizer = prepare_datasets(train_path, test_path)

lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

lstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy', f1_m])

print(lstm.summary())

lstm.fit(train_data, epochs=10, validation_data=test_data, class_weight={1:4, 0:1})
