import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

def load_data(file_path):
    return pd.read_csv(file_path)

def split_data(df):
    X = df.drop(columns=['Insect'])
    y = df['Insect']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=44)
    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train):
    model = DecisionTreeClassifier(random_state=44)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    joblib.dump(model, model_path)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Insect classification')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_train.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file):
    df = load_data(input_file)
    X_train, X_val, y_train, y_val = split_data(df)
    model = train_model(X_train, y_train)
    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)