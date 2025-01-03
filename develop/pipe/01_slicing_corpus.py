import os
import re
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, "../.."))))

from develop.utils.paths import DATA
from develop.utils.toolbox import ToolBox

catalog_path = "pg_catalog.csv"
input_dir = os.path.join(DATA, "00_pruned_corpus")
output_dir = os.path.join(DATA, "01_sliced_corpus")
output_all_dir = os.path.join(DATA, "01_sliced_corpus")

def prepare_csv(path):
    catalog = pd.read_csv(os.path.join(DATA, path))
        
    catalog["Issued"] = pd.to_datetime(catalog["Issued"])
    catalog["Issued"] = catalog["Issued"].dt.year
    catalog['Text#'] = "pg" + catalog['Text#'].astype(str)
    catalog = catalog[catalog["Language"] == "en"]
    
    print(catalog[['Text#', 'Title', 'Issued']].head())
    catalog.to_csv(os.path.join(DATA, ("01_" + catalog_path)), index=False)
    return catalog
    


def concatenate_files_by_year(df, input_dir, output_dir):
    """
    Concatenate text files based on their issuance year and save the result in the output directory.

    Parameters:
    - df: DataFrame containing metadata with 'Text' (file identifiers) and 'Issued' (year).
    - input_dir: Path to the directory containing the individual text files.
    - output_dir: Path to the directory where the concatenated files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    non_existent_files = 0
    non_admissible_files = 0

    grouped = df.groupby("Issued")["Text#"]
    valid_ids = set(df["Text#"])

    for year, file_ids in tqdm(grouped, desc="Concatenating files by year"):
        combined_text = ""
        
        for file_id in file_ids:
            if file_id in valid_ids:
                folder_id = re.search(r'\d+', file_id).group()
                file_path = os.path.join(input_dir, folder_id, file_id + ".txt")
                breakpoint()
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        combined_text += f.read() + "\n\n"  
                else:
                    non_existent_files += 1
            else:
                non_admissible_files += 1

        output_file = os.path.join(output_dir, f"combined_{year}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined_text)
        
    print(f"Number of non-existent files: {non_existent_files}")
    print(f"Number of non-admissible files: {non_admissible_files}")
    
def concat_all_txt(input_dir, output_dir):
    "Concatenate all the files in the input_dir into a single file in the output_dir incrementally"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "combined_all.txt")
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        tb = ToolBox()
        txt_files = list(tb.get_all_txt_files(input_dir))
        
        for file in tqdm(txt_files, desc="Concatenating files"):
            try:
                with open(file, "r", encoding="utf-8") as in_f:
                    content = in_f.read()
                    out_f.write(content + "\n\n")
            except Exception as e:
                print(f"Error reading {file}: {e}")

    print(f"Concatenated text for all files saved to {output_file}")
    
    
if __name__ == "__main__":
    df = prepare_csv(catalog_path)
    concatenate_files_by_year(df, input_dir, output_dir)
    # concat_all_txt(output_dir, output_all_dir)
    