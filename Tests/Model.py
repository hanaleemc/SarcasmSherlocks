import tensorflow as tf
import random
import pandas as pd
from transformers import DistilBertTokenizer
from Data_Preprocessing import preprocess_type_II, preprocess_type_III, \
    preprocess_type_I, preprocess_type_IV_A, preprocess_type_IV_B
from Evaluation import f1_m, balanced_accuracy
from keras import regularizers
from keras.metrics import AUC
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Dropout

#Add seed for weights initialization stability
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

#Choosing model type to train
def create_model(model_type, tokenizer):

    l2_regularizer = regularizers.l2(1e-4)

    if model_type == 'BiLSTM':
        model = tf.keras.Sequential([
            #Complex model
            tf.keras.layers.Embedding(len(tokenizer.vocab), 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')

            ##Complex model with L2 regul and droput
            # tf.keras.layers.Embedding(len(tokenizer.vocab), 64),
            # tf.keras.layers.Bidirectional(
            #     tf.keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=l2_regularizer,
            #                          dropout=0.2, recurrent_dropout=0.2)),
            # tf.keras.layers.TimeDistributed(
            #     tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_regularizer)),
            # tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.Bidirectional(
            #     tf.keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=l2_regularizer,
            #                          dropout=0.2, recurrent_dropout=0.2)),
            # tf.keras.layers.TimeDistributed(
            #     tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_regularizer)),
            # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, kernel_regularizer=l2_regularizer, dropout=0.2)),
            # tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=l2_regularizer),
            # tf.keras.layers.Dense(1, activation='sigmoid')

            # #Complex model with dropout
            # tf.keras.layers.Embedding(len(tokenizer.vocab), 64),
            # tf.keras.layers.Bidirectional(LSTM(32, return_sequences=True)),
            # tf.keras.layers.Dropout(0.3),  # Add dropout after the first LSTM layer
            # tf.keras.layers.TimeDistributed(Dense(64, activation='relu')),
            # tf.keras.layers.Dropout(0.3),  # Add dropout after the first TimeDistributed Dense layer
            # tf.keras.layers.Bidirectional(LSTM(32, return_sequences=True)),
            # tf.keras.layers.Dropout(0.3),  # Add dropout after the second LSTM layer
            # tf.keras.layers.TimeDistributed(Dense(64, activation='relu')),
            # tf.keras.layers.Dropout(0.3),  # Add dropout after the second TimeDistributed Dense layer
            # tf.keras.layers.Bidirectional(LSTM(32)),
            # tf.keras.layers.Dropout(0.3),  # Add dropout after the third LSTM layer
            # tf.keras.layers.Dense(8, activation='relu'),
            # tf.keras.layers.Dropout(0.3),  # Add dropout before the final Dense layer
            # tf.keras.layers.Dense(1, activation='sigmoid')

            # #Simpler model with dropout
            # tf.keras.layers.Embedding(len(tokenizer.vocab), 32),  # Reduced embedding dimension to 32
            # tf.keras.layers.Bidirectional(LSTM(16, return_sequences=True)),  # Reduced LSTM units and kept it bidirectional
            # tf.keras.layers.Dropout(0.3),  # Added dropout for regularization
            # tf.keras.layers.TimeDistributed(Dense(32, activation='relu')),  # Reduced units in the TimeDistributed Dense layer
            # tf.keras.layers.Dropout(0.3),  # Added another dropout layer for regularization
            # tf.keras.layers.Bidirectional(LSTM(16)),  # Reduced LSTM units in the final LSTM layer
            # tf.keras.layers.Dense(8, activation='relu'),  # Kept a single Dense layer with reduced units before the output layer
            # tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer remains the same

            # #Simpler model
            # Embedding(len(tokenizer.vocab), 32),  # Reduced embedding dimension to 32
            # Bidirectional(LSTM(16, return_sequences=True)),  # Reduced LSTM units and kept it bidirectional
            # TimeDistributed(Dense(32, activation='relu')),  # Reduced units in the TimeDistributed Dense layer
            # Bidirectional(LSTM(16)),  # Reduced LSTM units in the final LSTM layer
            # Dense(8, activation='relu'),  # Kept a single Dense layer with reduced units before the output layer
            # Dense(1, activation='sigmoid')  # Output layer remains the same

        ])


    elif model_type == 'LSTM':
        model = tf.keras.Sequential([
            
			#Complex model 
			# tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
            # tf.keras.layers.LSTM(32, return_sequences=True),
            # tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
            # tf.keras.layers.LSTM(32, return_sequences=True),
            # tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
            # tf.keras.layers.LSTM(32),
            # tf.keras.layers.Dense(8, activation='relu'),
            # tf.keras.layers.Dense(1, activation='sigmoid')
			
			#Complex model with dropout and regularization 
			#tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
    #         tf.keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=l2_regularizer, dropout=0.2,
    #                              recurrent_dropout=0.2),
    #         tf.keras.layers.TimeDistributed(
    #             tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2_regularizer)),
    #         tf.keras.layers.Dropout(0.5),
    #         tf.keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=l2_regularizer, dropout=0.2,
    #                              recurrent_dropout=0.2),
    #         tf.keras.layers.TimeDistributed(
    #             tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2_regularizer)),
    #         tf.keras.layers.LSTM(32, kernel_regularizer=l2_regularizer, dropout=0.2),
    #         tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=l2_regularizer),
    #         tf.keras.layers.Dense(1, activation='sigmoid')
			
			#Complex model with regularization 
	#		  tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
    #         tf.keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=l2_regularizer),
    #         tf.keras.layers.TimeDistributed(
    #             tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2_regularizer)),
    #         tf.keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=l2_regularizer),
    #         tf.keras.layers.TimeDistributed(
    #             tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2_regularizer)),
    #         tf.keras.layers.LSTM(32, kernel_regularizer=l2_regularizer),
    #         tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=l2_regularizer),
    #         tf.keras.layers.Dense(1, activation='sigmoid')

            ## Simplified model, less layers
            # tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
            # tf.keras.layers.LSTM(32),  # Removed return_sequences to reduce depth
            # # Removed one LSTM layer and associated TimeDistributed Dense layer to reduce complexity
            # tf.keras.layers.Dense(16, activation='relu'),  # Reduced the number of units in the Dense layer
            # tf.keras.layers.Dense(1, activation='sigmoid')

            # Simplified model, less layers, but with regularization
            # tf.keras.layers.Embedding(len(tokenizer.vocab), 32,
            #                           embeddings_regularizer=l2_regularizer),
            # # Added L2 regularization to Embedding layer
            # tf.keras.layers.LSTM(32, kernel_regularizer=l2_regularizer),  # Added L2 regularization to LSTM layer
            # tf.keras.layers.Dense(16, activation='relu',
            #                       kernel_regularizer=l2_regularizer),  # Added L2 regularization to Dense layer
            # tf.keras.layers.Dense(1, activation='sigmoid',
            #                       kernel_regularizer=l2_regularizer)  # Added L2 regularization to final Dense layer

            # Simplified model, less layers, but with dropout
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

    # # Specify the learning rate
    # custom_learning_rate = 0.001
    # #
    # # Instantiate the Adam optimizer with the custom learning rate
    # adam_optimizer = tf.keras.optimizers.Adam(learning_rate=custom_learning_rate)

    # #Compiling model with custom learning rate
    # model.compile(loss='binary_crossentropy', optimizer=adam_optimizer,
    #               metrics=['accuracy', f1_m, balanced_accuracy, AUC(name='aucroc')])
	
	# Compiling model with adam optimizer (adjusts learning rate automatically, starts with 0.001)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy', f1_m, balanced_accuracy, AUC(name='aucroc')])

    return model

