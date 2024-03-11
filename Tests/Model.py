import tensorflow as tf
import random
import pandas as pd
from transformers import DistilBertTokenizer
from Data_Preprocessing import preprocess_type_II, preprocess_type_III, \
    preprocess_type_I, preprocess_type_IV_A, preprocess_type_IV_B
from Evaluation import f1_m


def dataset_embedding(dataset_path, tokenizer, preproc_type=None, batch_size=32):

    dataset = pd.read_csv(dataset_path)[["tweet", "sarcastic"]]
    dataset = dataset[dataset['tweet'].notna()]

    # Conditional preprocessing based on preproc_type
    if preproc_type == 'PreprocI':
        dataset['tweet'] = dataset['tweet'].apply(preprocess_type_I)
    elif preproc_type == 'PreprocII':
        dataset['tweet'] = dataset['tweet'].apply(preprocess_type_II)
    elif preproc_type == 'PreprocIII':
        dataset['tweet'] = dataset['tweet'].apply(preprocess_type_III)
    elif preproc_type == 'PreprocIV_A':
        dataset['tweet'] = dataset['tweet'].apply(preprocess_type_IV_A)
    elif preproc_type == 'PreprocIV_B':
        dataset['tweet'] = dataset['tweet'].apply(preprocess_type_IV_B)

    tokenized_tweets = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweet)) for tweet in dataset['tweet']]
    tweets_with_len = [[tweet, dataset['sarcastic'].iloc[i], len(tweet)] for i, tweet in enumerate(tokenized_tweets)]
    random.Random(42).shuffle(tweets_with_len)
    tweets_with_len.sort(key=lambda x: x[2])
    sorted_tweets_labels = [(tweet_lab[0], tweet_lab[1]) for tweet_lab in tweets_with_len]  # remove tweet len

    processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_tweets_labels, output_types=(tf.int32, tf.int32))

    return processed_dataset.padded_batch(batch_size, padded_shapes=((None,), ()))


def prepare_datasets(train_path, test_path, preproc_type):
    # replaced their tokenizer with distil bert-base-uncased
    model_name = 'distilbert-base-uncased'

    # Initialize the tokenizer and model specifically for DistilBERT
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    # model = TFDistilBertModel.from_pretrained(model_name)

    dataset_train = dataset_embedding(train_path, tokenizer, preproc_type)
    dataset_test = dataset_embedding(test_path, tokenizer, preproc_type)

    return dataset_train, dataset_test, tokenizer

#Choosing model type to train
def create_model(model_type, tokenizer):
    if model_type == 'BiLSTM':
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    elif model_type == 'LSTM':
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])

    return model

