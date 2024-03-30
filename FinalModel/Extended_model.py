import tensorflow as tf
import random
from Evaluation import f1_m, balanced_accuracy
from keras import regularizers
from keras.metrics import AUC
from keras.optimizers import Adam

#Add seed for weights initialization stability
tf.random.set_seed(42)
random.seed(42)

# Choosing model type to train
def create_ext_model(model_type, tokenizer, learning_rate=0.001):

    # Creating an L2 regularization object with a regularization factor of 1e-4
    l2_regularizer = regularizers.l2(1e-4)

    if model_type == 'BiLSTM_L2_D':
        model = tf.keras.Sequential([
            #Complex model with L2 regularization and droput
            tf.keras.layers.Embedding(len(tokenizer.vocab), 64),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=l2_regularizer,
                                     dropout=0.2, recurrent_dropout=0.2)),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_regularizer)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=l2_regularizer,
                                     dropout=0.2, recurrent_dropout=0.2)),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_regularizer)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, kernel_regularizer=l2_regularizer, dropout=0.2)),
            tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    elif model_type == 'BiLSTM_D':
        model = tf.keras.Sequential([
            #Complex model with dropout
            tf.keras.layers.Embedding(len(tokenizer.vocab), 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    elif model_type == 'BiLSTM_simple':
        model = tf.keras.Sequential([
            #Simpler model, less layers
            tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    elif model_type == 'BiLSTM_simple_D':
        model = tf.keras.Sequential([
            #Simpler model with dropout
            tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    elif model_type == 'LSTM_L2_D':
        model = tf.keras.Sequential([
            # Complex model with dropout and regularization
            tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
                    tf.keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=l2_regularizer, dropout=0.2,
                                         recurrent_dropout=0.2),
                    tf.keras.layers.TimeDistributed(
                        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2_regularizer)),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=l2_regularizer, dropout=0.2,
                                         recurrent_dropout=0.2),
                    tf.keras.layers.TimeDistributed(
                        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2_regularizer)),
                    tf.keras.layers.LSTM(32, kernel_regularizer=l2_regularizer, dropout=0.2),
                    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=l2_regularizer),
                    tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    elif model_type == 'LSTM_D':
        model = tf.keras.Sequential([
            # Complex model with dropout
            tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
            tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
            tf.keras.layers.LSTM(32, dropout=0.2),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    elif model_type == 'LSTM_simple':
        model = tf.keras.Sequential([
            # Simpler model, less layers
            tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    elif model_type == 'LSTM_simple_D':
        model = tf.keras.Sequential([
            # Simpler model with dropout
            tf.keras.layers.Embedding(len(tokenizer.vocab), 32),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Applying the learning rate to the Adam optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Compiling the model
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', f1_m, balanced_accuracy, AUC(name='aucroc')])

    return model