from pathlib import Path
from typing import Dict, List

import pandas as pd

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_and_prepare_data(file_path, var_name, weights, plot_group_by):
    """Read and clean data."""
    df = pd.read_csv(file_path)

    # Ensure columns are numeric and handle NaNs
    df[var_name] = pd.to_numeric(df[var_name], errors="coerce")
    df[weights] = pd.to_numeric(df[weights], errors="coerce")

    # Filter rows where exposure > 0 and drop NaNs
    df = df[df[var_name] > 0]
    df.dropna(subset=[var_name, weights, plot_group_by], inplace=True)

    return df


def get_image_paths(directory: Path, pattern: str = "*.[pP][nN][gG]") -> List[Path]:
    """Retrieve all image paths from the given directory matching the pattern."""
    return sorted(directory.glob(pattern), key=lambda p: p.name.lower())


def extract_title_from_summary(summary: str) -> str:
    """Extract a descriptive title from the summary text, accommodating keyword variations."""
    # Mapping of various keyword variations to a standard title
    keyword_map = {
        "PSI": ["psi", "PSI"],
        "HHI": ["hhi", "HHI"],
        "Migration Matrix": [
            "migration matrix",
            "Migration Matrix",
            "migration_matrix",
            "Migration-Matrix",
        ],
        "Discriminatory Power": [
            "discriminatory power",
            "Discriminatory Power",
            "discriminatory_power",
            "Discriminatory-Power",
            "discriminatory",
            "somers_d",
            "Somers_d",
            "Somers_D",
            "Somer's D",
            "Somer's d",
        ],
        "Distribution": [
            "distribution",
            "Distribution",
            "distribution_chart",
            "Distribution-Chart",
        ],
    }

    # Loop through the keyword map to find appropriate matches
    for standard_keyword, variations in keyword_map.items():
        for variation in variations:
            if variation in summary:
                return f"{standard_keyword} Chart Summary"

    return "Chart Summary"  # Default title if no matches are found.


def create_sample_func(
    file_path: str, file_name: str, date_column_name: str, date_for_split: str
) -> Dict[str, pd.DataFrame]:
    """
    Create a sample by splitting the dataset based on a given date. The function reads a CSV file, parses the date
    column, splits the data based on the provided date, assigns a 'sample_id' to each row, and saves three separate
    samples: 'old', 'new', and 'all'.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to read.
    file_name : str
        The base name for the CSV files to be saved.
    date_column_name : str
        The name of the column in the DataFrame containing date values to use for the split.
    date_for_split : str
        The date string used to split the dataset.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary with keys 'sample_old', 'sample_new', and 'sample_all', each associated with a DataFrame
        representing different parts of the dataset.
    """

    try:
        df = pd.read_csv(file_path)
        df[date_column_name] = pd.to_datetime(df[date_column_name], errors="raise")
        date_for_split_dt = pd.to_datetime(date_for_split, errors="raise")

        df["sample_id"] = df[date_column_name].apply(
            lambda x: "new" if x > date_for_split_dt else "old"
        )

        sample_dict = {
            "sample_old": df[df["sample_id"] == "old"],
            "sample_new": df[df["sample_id"] == "new"],
            "sample_all": df,
        }

        # Save samples
        for key, dataframe in sample_dict.items():
            smpl_filename = f"{file_name}_{key}.csv"
            output_path = OUTPUT_DIR / smpl_filename
            dataframe.to_csv(output_path, index=False)

        return sample_dict
    except Exception as e:
        raise RuntimeError(f"Error during sampling: {e}")
