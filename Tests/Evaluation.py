from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf

# Functions for evaluation
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
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def balanced_accuracy(y_true, y_pred):
    recall_positive = recall_m(y_true, y_pred)
    recall_negative = recall_m(1 - y_true, 1 - y_pred)
    balanced_acc = (recall_positive + recall_negative) / 2
    return balanced_acc


#Functions for plotting the results
def plot_history(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['f1_m'], label='Training F1')
    plt.plot(history.history['val_f1_m'], label='Validation F1')
    plt.title('F1 over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()

    plt.tight_layout()
    plt.show()