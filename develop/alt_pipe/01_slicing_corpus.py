import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, "../.."))))
from develop.utils.paths import DATA_ALT
from develop.utils.logger import LoggerManager

logger = LoggerManager(
    name = "slicer", 
    log_file = "slicer.log"
    ).get_logger()

def load_data(file_path: str) -> pd.DataFrame:
    """Load parquet DataFrame.

    Args:
        file_path (str): File of the parquet

    Returns:
        pd.DataFrame: DataFrame
    """
    logger.info(f"Loading {file_path}")
    return pd.read_parquet(file_path)

def plot_year_distribution(df: pd.DataFrame) -> None:
    """Plot the distribution of articles by year.

    Args:
        df (pd.DataFrame): DataFrame
    """
    df["year"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Year")
    plt.ylabel("Article Count")
    plt.title("Distribution of Articles by Year")
    plt.show()
    logger.info("Plotting articles by year of publication.")

def preprocess_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Merges the columns of title and excerpt.

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: DataFrame
    """
    df = df.copy()
    df["corpus"] = df["title"] + " " + df["excerpt"]
    logger.info("Preprocessing done.")
    return df


def save_corpus_by_year(df: pd.DataFrame, output_folder: str) -> None:
    """Aggregates articles based on year of publication and saves them.

    Args:
        df (pd.DataFrame): _description_
        output_folder (str): _description_
    """
    os.makedirs(output_folder, exist_ok=True)
    for year, df_year in df.groupby("year"):
        output_file = os.path.join(output_folder, f"{year}.txt")
        df_year["corpus"].to_csv(output_file, index=False, header=False)
    logger.info(f"Saved {df['year'].nunique()} corpus files in {output_folder}")


def main() -> None:
    """Triggers the pipeline.
    """
    input_path = os.path.join(DATA_ALT, "nyt_data.parquet")
    output_folder = os.path.join(DATA_ALT, "02_sliced_corpus")
    os.makedirs(output_folder, exist_ok=True)

    df = load_data(input_path)
    plot_year_distribution(df)
    df = preprocess_corpus(df)
    save_corpus_by_year(df, output_folder)


if __name__ == "__main__":
    main()
