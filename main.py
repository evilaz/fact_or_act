"""Contains a full workflow with the functionality: building the model, saving it, testing it and making an
inference for a random test sample. """

import json

import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

from services import MultiLabelOneHotEncoder, CustomOneHotEncoder
from services import TextPreProcessor
from utils import labels_to_binary, get_random_row_json_from_test_data, json_to_dataframe

# To ensure that the test samples will not be part of the train samples
split_random_state = 5


def build_model_from_data(data_filename: str = "data.csv", pipe_filename="pipeline.joblib"):
    """Build a model from the data, save the pipeline and validates the model with the test dataset split."""

    df = pd.read_csv(data_filename, index_col="id")

    # Convert the problem to binary classification
    df["label"] = df["label"].apply(labels_to_binary)

    x = df[[col for col in df.columns if col != "label"]]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=split_random_state)

    preprocessor = TextPreProcessor()

    encoder = ColumnTransformer(
        transformers=[
            ("one_hot_encoder", CustomOneHotEncoder(), "party_affiliation"),
            ("one_hot_encoder_multi", MultiLabelOneHotEncoder(), "subject"),
            ("tfidf_context", TfidfVectorizer(max_features=120), "rest_context"),
            ("tfidf_statement", TfidfVectorizer(max_features=250), "statement"),
        ],
        remainder="passthrough",
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("encoder", encoder),
            ("scaler", MaxAbsScaler()),
            ("classifier", SVC(C=0.2)),
        ]
    )

    # Apply the transformations and build the model
    pipe.fit(x_train, y_train)

    # Validation on test set
    y_pred = pipe.predict(x_test)

    print("Results:")
    print(classification_report(y_test, y_pred))

    # The pipeline will be saved so that it can be used directly on new data
    dump(pipe, pipe_filename)

    return None


def predict_statement_truthfulness(statement_json, pipe_filename="pipeline.joblib") -> bool:
    """Predict whether the input json statement is truthful or not using the saved model (pipeline)."""

    x_test = json_to_dataframe(statement_json)

    # Load the pipeline and predict the class of x_test
    loaded_pipe = load(pipe_filename)
    prediction = loaded_pipe.predict(x_test)

    return prediction[0]


def main():
    # Build and save the model (it takes some time)
    print("Building the model...hang in there..")
    build_model_from_data("data.csv")

    print("Model build and pipeline saved. Time to predict a random sample.\n")

    # Get a random test row from the unseen data and make a prediction
    test_row_json = get_random_row_json_from_test_data(
        "data.csv", random_state=split_random_state, verbose=True
    )
    test_row = json.loads(test_row_json)
    print("Statement of the random test sample is: ")
    print(test_row["statement"])
    print(f'Speaker "{test_row["speaker"]}" at {test_row["context"]}\n')

    prediction = predict_statement_truthfulness(test_row_json)

    print(f"Current test sample was predicted as: {prediction}.")


if __name__ == "__main__":
    main()
