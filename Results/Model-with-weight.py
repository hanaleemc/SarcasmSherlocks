import tensorflow as tf
import random
import pandas as pd
from transformers import DistilBertTokenizer
from Data_Preprocessing import preprocess_type_II, preprocess_type_III, preprocess_type_I, preprocess_type_IV_A, preprocess_type_IV_B
from Evaluation import f1_m, balanced_accuracy
from keras import regularizers
from keras.metrics import AUC
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Dropout
from keras.losses import binary_crossentropy

# Add seed for weights initialization stability
tf.random.set_seed(42)

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

def weighted_binary_cross_entropy_with_logits(pos_weight):
    def loss_func(labels, logits):
        labels = tf.cast(labels, tf.float32)  # Cast labels to float32
        return tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=tf.cast(pos_weight, tf.float32))
    return loss_func

# Choosing model type to train
def create_model(model_type, tokenizer, loss_function, pos_weight=None):

    l2_regularizer = regularizers.l2(1e-4)

    if model_type == 'BiLSTM':
        model = tf.keras.Sequential([
            # Complex model
            tf.keras.layers.Embedding(len(tokenizer.vocab), 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    elif model_type == 'LSTM':
        model = tf.keras.Sequential([
            # Complex model with dropout
            tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
            tf.keras.layers.Dropout(0.3),  # Add dropout after the embedding layer
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.3),  # Add dropout after the LSTM layer
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),  # Add dropout after the Dense layer
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    if loss_function == 'binary_crossentropy':
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy', f1_m, balanced_accuracy, AUC(name='aucroc')])
    elif loss_function == 'weighted_binary_cross_entropy':
        if pos_weight is None:
            raise ValueError("The pos_weight argument is required when using weighted_binary_cross_entropy loss.")
        model.compile(loss=weighted_binary_cross_entropy_with_logits(pos_weight=pos_weight), optimizer='adam',
                      metrics=['accuracy', f1_m, balanced_accuracy, AUC(name='aucroc')])
    else:
        raise ValueError("Unsupported loss function: {}".format(loss_function))

    return model
