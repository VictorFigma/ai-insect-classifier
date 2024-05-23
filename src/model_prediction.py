import pandas as pd
import argparse
import joblib
import json

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.abs()
    df = df.round(2)
    df.loc[:, 'Hour'] *= 60  # Adapt to trained data format
    df.loc[:, 'Minutes'] = 0
    return df

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def make_predictions(df, model):
    predictions = model.predict(df)
    return predictions

def save_predictions(predictions, predictions_file):
    output = {'target': {str(k): int(v) for k, v in enumerate(predictions)}}
    with open(predictions_file, 'w') as f:
        json.dump(output, f)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Insect classification')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/test.csv', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json', 
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    df = load_data(input_file)
    model = load_model(model_file)
    predictions = make_predictions(df, model)
    save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)