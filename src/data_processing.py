import argparse
import pandas as pd
from scipy import stats

def load_data(file_path):
    return pd.read_csv(file_path)

# Handle outliers
def clean_data(df):
    z_scores = stats.zscore(df)
    abs_z_scores = abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df_clean = df[filtered_entries]
    return df_clean

def preprocess_data(df):
    df = df.abs()
    df = df.round(2)
    df.loc[:, 'Hour'] *= 60 # Insect hour of activity (sleep cycle) is much more important than minutes
    df.loc[:, 'Minutes'] = 0
    return df 

def save_data(df, output_file):
    df.to_csv(output_file, index=False)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Insect classification')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/train.csv',
        help='Path to train.csv'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_train.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    df = load_data(input_file)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)