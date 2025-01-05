from functools import reduce
import os
import pandas as pd
from collections import Counter

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, "../.."))))
from develop.utils.paths import DATA

def compute_word_frequencies(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = f.read().split()
    return Counter(words)

def create_combined_df(word_freq_dfs):
    combined_df = reduce(
        lambda left, right: 
            pd.merge(left, right, on='Word', how='outer'), 
            word_freq_dfs
            )
    combined_df.fillna(0, inplace=True)
    combined_df['Total'] = combined_df.sum(axis=1)
    combined_df.sort_values(by='Total', ascending=False, inplace=True)
    combined_df = combined_df[['Total'] + [col for col in combined_df.columns if col != 'Total']]
    combined_df = filter_low_frequency(combined_df, 'Total', 10)    
    return combined_df

def filter_low_frequency(df, column, threshold):
    return df[df[column] > threshold]

def process_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    word_freq_dfs = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            word_frequencies = compute_word_frequencies(file_path)
            
            freq_df = pd.DataFrame(list(word_frequencies.items()), columns=['Word', 'Frequency'])
            freq_df.set_index('Word', inplace=True)
            file_name = file_name[:-4]
            freq_df.sort_values(by='Frequency', ascending=False, inplace=True)            
            freq_df.rename(columns={'Frequency': file_name}, inplace=True)
            word_freq_dfs.append(freq_df)

    combined_df = create_combined_df(word_freq_dfs)
    combined_df.to_csv(os.path.join(output_folder, 'word_frequencies.csv'))

if __name__ == "__main__":
    folder_path = os.path.join(DATA, "02_preprocessed_corpus")
    output_folder = os.path.join(DATA, "04_word_frequencies")
    process_folder(folder_path, output_folder)
