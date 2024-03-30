import tensorflow as tf
import pandas as pd
from transformers import DistilBertTokenizer
from Data_Preprocessing import preprocess_type_II, preprocess_type_III, \
    preprocess_type_I, preprocess_type_IV_A, preprocess_type_IV_B
from Evaluation import f1_m, balanced_accuracy
from keras.metrics import AUC
import nltk
from textblob import TextBlob
from keras.optimizers import Adam

tf.random.set_seed(42)

nltk.download('vader_lexicon')

def get_sentiment_polarity(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

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

    #Check if the right preprocessing type is chosen
    print(f"Processing type is:", preproc_type)

    #Add a column for sentiment polarity
    dataset['sentiment'] = dataset['tweet'].apply(get_sentiment_polarity)

    tokenized_tweets = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweet)) for tweet in dataset['tweet']]
    # tweets_with_len = [[tweet, dataset['sarcastic'].iloc[i], len(tweet)] for i, tweet in enumerate(tokenized_tweets)]
    tweets_with_len = [[tweet, dataset['sarcastic'].iloc[i], len(tweet), dataset['sentiment'].iloc[i]] for i, tweet in
                       enumerate(tokenized_tweets)]

    def gen():
        for tweet, sarcastic, _, sentiment in tweets_with_len:
            yield {"text_input": tweet, "sentiment_input": [sentiment]}, sarcastic

    # Adjust the output_types and output_shapes to match the structure of the generated items
    processed_dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=({"text_input": tf.int32, "sentiment_input": tf.float32}, tf.int32),
        output_shapes=(
        {"text_input": tf.TensorShape([None]), "sentiment_input": tf.TensorShape([1])}, tf.TensorShape([]))
    )

    return processed_dataset.padded_batch(batch_size,
                                          padded_shapes=({"text_input": (None,), "sentiment_input": (1,)}, ()))


def prepare_datasets_polar(train_path, test_path, preproc_type):

    model_name = 'distilbert-base-uncased'

    # Initialize the tokenizer and model specifically for DistilBERT
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    dataset_train = dataset_embedding(train_path, tokenizer, preproc_type)
    dataset_test = dataset_embedding(test_path, tokenizer, preproc_type)

    return dataset_train, dataset_test, tokenizer

#Choosing model type to train
def create_model_polarity(model_type, tokenizer, learning_rate=0.001):

    if model_type == 'BiLSTM_polar':

        # Define the input layer for the text
        text_input = tf.keras.Input(shape=(None,), dtype='int32', name='text_input')
        # Define input layer for the sentiment polarity scores
        sentiment_input = tf.keras.Input(shape=(1,), name='sentiment_input')

        #Passing input through layers
        x = tf.keras.layers.Embedding(len(tokenizer.vocab), 64)(text_input)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu'))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu'))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)

        # Concatenate the sentiment input with the LSTM output
        concatenated = tf.keras.layers.concatenate([x, sentiment_input], axis=-1)

        # A dense layer with relu activation following the concatenated inputs
        x = tf.keras.layers.Dense(8, activation='relu')(concatenated)
        # The output layer with a sigmoid activation for binary classification
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        #Constructing the model
        model = tf.keras.Model(inputs=[text_input, sentiment_input], outputs=output)

    elif model_type == 'LSTM_polar':

        text_input = tf.keras.Input(shape=(None,), dtype='int32', name='text_input')
        sentiment_input = tf.keras.Input(shape=(1,), name='sentiment_input')

        x = tf.keras.layers.Embedding(len(tokenizer.vocab), 32)(text_input)
        x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu'))(x)
        x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu'))(x)
        x = tf.keras.layers.LSTM(32)(x)

        concatenated = tf.keras.layers.concatenate([x, sentiment_input], axis=-1)

        x = tf.keras.layers.Dense(8, activation='relu')(concatenated)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=[text_input, sentiment_input], outputs=output)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Applying the learning rate to the Adam optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Compiling the model
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', f1_m, balanced_accuracy, AUC(name='aucroc')])

    return model

