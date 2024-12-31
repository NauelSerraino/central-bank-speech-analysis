import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm

"""
Prune the corpus by removing direct references to the Gutenberg Project.
"""

source_dir = "data/txt-files.tar/txt-files/cache/epub"
target_dir = "data/txt-files.tar/txt-files/cache/epub-pruned"

Path(target_dir).mkdir(parents=True, exist_ok=True)

start_pattern = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*\*\*\*")
end_pattern = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*\*\*\*")
exception_pattern = re.compile(r"\*\*Welcome To The World of Free Plain Vanilla Electronic Texts\*\*")

correct_executions = 0
exception_files = 0
error_files = 0

def process_file(file_path, target_file_path):
    global correct_executions, exception_files, error_files

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if exception_pattern.search(content):
            exception_files += 1
            # print(f"Skipping {file_path} due to exception pattern.")
            return  
        
        start_match = start_pattern.search(content)
        end_match = end_pattern.search(content)
        if start_match and end_match:
            pruned_content = content[start_match.end():end_match.start()].strip()
        else:
            pruned_content = content

        os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
        with open(target_file_path, 'w', encoding='utf-8') as pruned_file:
            pruned_file.write(pruned_content)

        correct_executions += 1

    except Exception as e:
        error_files += 1
        # print(f"Error processing {file_path}: {e}")

def get_all_txt_files(directory):
    for root, dirs, files in os.walk(directory):
        dirs.sort(key=lambda x: int(x) if x.isdigit() else x)
        for file in files:
            if file.endswith('.txt'):
                yield os.path.join(root, file)

def process_all_files(limit=None):
    txt_files = list(get_all_txt_files(source_dir))
    if limit:
        txt_files = txt_files[:limit]
    for file in tqdm(txt_files, desc="Copying files"):
        process_file(file, os.path.join(target_dir, os.path.relpath(file, source_dir)))
        
def summarize_results():
    print(f"Correct executions: {correct_executions}")
    print(f"Files skipped due to exception pattern: {exception_files}")
    print(f"Files with errors: {error_files}")

if __name__ == "__main__":
    test_run = False
    if test_run:
        process_all_files(limit=10)
        summarize_results()
    else:
        process_all_files()
        summarize_results()