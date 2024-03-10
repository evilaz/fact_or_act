""" Generic utility methods. """

import json
import random

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split


def json_to_dataframe(json_sample):
    """Convert a json of a single statement data point to a 1-row Dataframe"""
    json_dict = json.loads(json_sample)
    df = pd.DataFrame(json_dict, index=[0])
    df = df.map(lambda x: np.nan if x is None else x)

    return df


def labels_to_binary(label: str) -> bool:
    """Turn the 6 labels to binary true/false.

    Specifically:
        "pants-fire", "false", "barely-true" ->  False
        "half-true", "mostly-true", "true" ->  True
    """

    if label.lower() in ["false", "pants-fire", "barely-true"]:
        return False
    else:
        return True


def perform_rfe(x_df, x_data, y_data, num_features=1, model=RandomForestClassifier()):
    """Runs RFE on a dataset using Logistic Regression to rank the features.

    x_df is the dataset including feature names, x_data, y_data are the train data as arrays.
    """

    # Create an RFE selector
    rfe = RFE(model, n_features_to_select=num_features)

    # Fit the RFE selector to your data
    rfe.fit(x_data, y_data)

    # Create a DataFrame to display the selected features, rankings, and names
    selected_features_df = pd.DataFrame(
        {"Feature_Name": x_df.columns, "Selected": rfe.support_, "Ranking": rfe.ranking_}
    )
    # Sort the DataFrame by ranking (most to least important)
    selected_features_df = selected_features_df.sort_values(by="Ranking")

    return selected_features_df


def get_features_names_from_pipeline(pipe, step=0):
    """Get feature names from encoding transformers, useful for feature selection and more interpretability.
    Args:
        pipe: sklearn Pipeline
        step: the step of column transformer in the pipeline
    """

    # preprocessor column transformer
    col_trn = pipe.steps[step][1]
    num_transformers = len(col_trn.transformers_)
    features_names = []
    for i in range(num_transformers):
        current_features = col_trn.transformers_[i][1].get_feature_names_out()
        print(
            f"Transformer '{col_trn.transformers_[i][0]}' created {len(current_features)} features. "
            f"e.g. {current_features[0:3]}."
        )

        if isinstance(current_features, list):
            features_names += current_features
        else:
            features_names += current_features.tolist()

    return features_names


def get_random_row_json_from_test_data(data_filename: str, random_state: int, verbose: bool = False):
    """Gets the entire dataset and the seed the train/test was split with and returns a random row from
    the unseen test data as a json object.
    """

    df = pd.read_csv(data_filename, index_col="id")
    x = df[[col for col in df.columns if col != "label"]]
    y = df["label"]

    # split data the same way as with training
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.25, random_state=random_state)

    random_i = random.sample(x_test.index.to_list(), k=1)[0]
    rand_test_statement_json = x_test.loc[random_i].to_json()

    if verbose:
        print(f"Test item has index {random_i} and true label: {y_test.loc[random_i]}")

    return rand_test_statement_json


def get_unique_subjects(df):
    """Flatten 'subjects' column and return the unique values (subjects)."""

    subjects_flattened = []
    _ = [subjects_flattened.extend(x.split(",")) for x in df["subject"].unique()]
    unique_subjects = set(subjects_flattened)

    return unique_subjects


def download_datasets(dir_name="data_files"):
    """Download the 3 datasets (train, test, split) to specified data directory."""

    dataset = load_dataset("liar")
    for subset in ["train", "test", "validation"]:
        dataset[subset].to_csv(f"{dir_name}/{subset}_ov.csv")  # ov: original version of dataset


def decode_label(label: int) -> str:
    """Decode numbered label to named label for readability."""

    decode_map = {5: "pants-fire", 0: "false", 4: "barely-true", 1: "half-true", 2: "mostly-true", 3: "true"}

    return decode_map[label]


def format_datasets(data_dir="data_files"):
    """Format the original datasets:
        - remove the credit count columns
        - decode labels (e.g. 2 -> 'mostly-true' see decode_label())
        - format id.json->id and set as index
    and write to csv files.
    """

    for dataset in ["train", "test", "validation"]:
        df = pd.read_csv(f"{data_dir}/{dataset}_ov.csv")

        df["id"] = df["id"].str.strip(".json")
        df.set_index(keys="id", drop=True, inplace=True)

        credit_counts_cols = [
            "barely_true_counts",
            "false_counts",
            "half_true_counts",
            "mostly_true_counts",
            "pants_on_fire_counts",
        ]
        df.drop(columns=credit_counts_cols, inplace=True)

        # Decode labels
        df["label"] = df["label"].apply(decode_label)

        # Write to new csv files
        df.to_csv(f"{data_dir}/{dataset}.csv")
