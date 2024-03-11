import argparse

from Model import prepare_datasets, create_model
from Evaluation import plot_history

train_path = './Data/train.En.csv'
test_path = './Data/test.En.csv'

def main():

    parser = argparse.ArgumentParser(description='Training LSTM or BiLSTM model')

    parser.add_argument('--pptype', dest='pptype',
                        choices=['PreprocI','PreprocII', 'PreprocIII', 'PreprocIV_A', 'PreprocIV_B'],
                        help='{PreprocI, PreprocII, PreprocIII, PreprocIV_A, PreprocIV_B}',
                        type=str, default='None')

    parser.add_argument('--model', dest='model_type',
                        choices=['LSTM', 'BiLSTM'],
                        help='{LSTM, BiLSTM}',
                        type=str, default='LSTM')

    args = parser.parse_args()

    # Pass the pptype argument to prepare_datasets
    train_data, test_data, tokenizer = prepare_datasets(train_path, test_path, args.pptype)

    # Create model based on the selected model type
    model = create_model(args.model_type, tokenizer)

    print(model.summary())

    #saving the metrics results
    results_visual = model.fit(train_data, epochs=10, validation_data=test_data, class_weight={1: 4, 0: 1})

    #plotting the metrics results
    plot_history(results_visual)

if __name__ == "__main__":
    main()


