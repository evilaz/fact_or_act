""" Contains custom service objects and the custom Transformers of the project."""

import re
from collections import Counter
from typing import Tuple

import nltk
import numpy as np
import pandas as pd
import spacy
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class TextHandler:
    """Wrapper for all text processing methods."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["tokenizer", "parser", "ner", "textcat"])

        # Download NLTK stop words (if not already downloaded) and store as an instance attribute
        nltk.download("stopwords", quiet=True)
        self.stopwords = set(stopwords.words("english"))

        nltk.download("punkt", quiet=True)
        self.punkt_tokenizer = nltk.tokenize.PunktSentenceTokenizer()

    @staticmethod
    def tokenize_text(text):
        return word_tokenize(text)

    @staticmethod
    def preprocess_numbers(text: str) -> str:
        """Preprocess numbers in the text and unifies the format."""

        # Normalize percentage  by replacing the % with the word percent
        text = re.sub(r"(\d+(\.\d+)?)%", r"\1 percent", text)

        # Normalize large numbers without commas by removing the comma 2,000 -> 20000
        text = re.sub(
            r"(\d{1,3}(?:,\d{3})*)(?:\.\d+)?",
            lambda x: x.group(1).replace(",", ""),
            text,
        )

        # Normalize decimals by replacing them with the integer part only
        text = re.sub(r"\b(\d+)\.\d+\b", r"\1", text)

        return text

    @staticmethod
    def preprocess_text_simple(text):
        """Light text processing, applies only lowercasing and breaks hyphen-joined words"""

        if not isinstance(text, str):
            return text

        # Lowercase and tokenize
        text = text.lower()
        tokens = word_tokenize(text)

        # Retain words that are joined with hyphen, split in two without hyphen
        tokens = [subtoken for token in tokens for subtoken in token.split("-")]

        return " ".join(tokens)

    def preprocess_text(
        self,
        text: str,
        keep_punctuation: bool = False,
        stop_words_basic: bool = True,
    ) -> str:
        """Text preprocessing with different options about removing stopping words and punctuation."""

        tokens = word_tokenize(text)

        # Retain words that are joined with hyphen to keep the meaning e.g. ex-smoker, non-smoker
        tokens = [subtoken for token in tokens for subtoken in token.split("-")]

        # Handle with punctuation
        if keep_punctuation:
            tokens = [word for word in tokens if word.isalnum() or word in [".", "%", "-", "?", "!"]]
        else:
            # Removing special characters and punctuation
            tokens = [word for word in tokens if word.isalnum()]
            # tokens = [word for word in tokens if word.isalnum() or word.__contains__("-")] # to keep ex-smoker as 1

        # Stopword removal, limited or normal
        if stop_words_basic:
            stop_words = {"the", "this", "a", "an", "and", "that"}
        else:
            stop_words = self.stopwords
        tokens = [word for word in tokens if word.lower() not in stop_words]

        return " ".join(tokens)

    @staticmethod
    def apply_lemmatization_after_processing(text: str):
        """Lemmatization. Not used atm."""
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    @staticmethod
    def remove_repetition_and_extra_whitespaces(text: str):
        """Remove extra whitespaces and repeated words that might have occured while columns were concatenated.
        e.g. "chain email email"
        """
        tokens = word_tokenize(text)
        unique_tokens = list(dict.fromkeys(tokens))
        return " ".join(unique_tokens)

    def pos_counts(self, text: str) -> dict[str, int]:
        """Count the Part Of Speech (POS) to possibly use as features."""

        doc = self.nlp(text)

        all_pos_counts = Counter([token.pos_ for token in doc])

        # Initialize dictionary only for the pos to use
        pos_counts = dict.fromkeys(["verb", "adj", "adv", "noun", "num", "aux", "cconj", "pron"])

        # Fill with values found
        pos_counts = {f"{key}_num": all_pos_counts[key.upper()] for key in pos_counts.keys()}

        return pos_counts

    def pos_perc(self, text: str) -> dict[str, int]:
        """Count the Part Of Speech (POS) ratios to length of sentence. (Not used currently)."""

        doc = self.nlp(text)
        all_pos_counts = Counter([token.pos_ for token in doc])

        # Initialize dictionary only for the pos to use
        pos_counts = dict.fromkeys(["verb", "adj", "adv", "noun", "num", "aux", "cconj", "pron"])

        # Fill with values found
        text_length = len(doc)
        pos_counts = {
            f"{key}_perc": round(all_pos_counts[key.upper()] / text_length, 2) for key in pos_counts.keys()
        }

        return pos_counts

    @staticmethod
    def count_tokens_and_avg_length(text: str) -> Tuple[int, float]:
        """Counts num of tokens and average length of token combined."""
        tokens = word_tokenize(text)
        avg_token_length = round(sum([len(token) for token in tokens]) / len(tokens), 2)
        return len(tokens), avg_token_length

    @staticmethod
    def count_tokens(text):
        """Count the number of tokens"""
        return len(word_tokenize(text))

    @staticmethod
    def avg_token_length(text):
        """Calculate the average token length in the sentence"""
        tokens = word_tokenize(text)
        avg_token_length = round(sum([len(token) for token in tokens]) / len(tokens), 2)
        return avg_token_length

    @staticmethod
    def replace_numbers(text):
        """Replace numbers with a d- string."""

        def repl(match):
            num = int(match.group(0))
            if num <= 1000:
                return "d" * len(str(num))
            else:
                return "dddd"

        return re.sub(r"\b\d+\b", repl, text)


#  -------   Transformers  ---------


class TextPreProcessor(BaseEstimator, TransformerMixin):
    """This Transformer applies text processing and extracts some text features from the main text (statement).
    It's tailored to the data of the specific dataset.

    Specifically:
    1. It applies some basic preprocessing (lowercasing, numbers cleanup)
    2. Extract some POS features from the column "statement"
    3. Applies further preprocessing (removing stop words, punctuation etc.)
    4. Appends the processed columns with the new features
    5. Concatenates some of the columns into one

    """

    def __init__(self, stop_words_basic=False):
        self.text_handler = TextHandler()
        self.feature_extractor = StatementFeatureExtractor()
        self.feature_names = []
        self.final_columns = []
        self.stop_words_basic: bool = stop_words_basic
        self.apply_lemmatization: bool = False
        self.keep_punctuation: bool = False
        self.exclude_columns = ["party_affiliation"]
        self.nan_replacement_value = ""

    @staticmethod
    def _preprocess_numbers_in_statement(df):
        """Standardisation and clean up of numbers in the 'statement' column.
        Following processing is applied:
        - % to percent
        - decimals -> integer part only
        - remove commas 25,000 -> 25000

        It is replacing text_handler.preprocess_numbers + apply because it's faster.
        """
        df_ = df.copy()
        df_["statement"] = df_["statement"].str.replace(r"(\d+(\.\d+)?)%", r"\1 percent", regex=True)

        # Normalize large numbers without commas by removing the comma 2,000 -> 20000
        df_["statement"] = df_["statement"].str.replace(
            r"(\d{1,3}(?:,\d{3})*)(?:\.\d+)?", lambda x: x.group(1).replace(",", ""), regex=True
        )

        # Normalize decimals by replacing them with the integer part only
        df_["statement"] = df_["statement"].str.replace(r"\b(\d+)\.\d+\b", r"\1", regex=True)

        return df_

    def shallow_preprocessing(self, df):
        """Basic preprocessing, keeping the sentence still as intact as possible for POS extraction."""

        df_ = df.copy()

        # preprocess numbers (for 'statement' column only)
        df_ = self._preprocess_numbers_in_statement(df_)
        # df_["statement"] = df_["statement"].apply(self.text_handler.preprocess_numbers)

        # Clean unicode
        df_["statement"] = df_["statement"].str.replace(r"[\u0080-\uffff]", "", regex=True)

        # Remove the prefix 'says' ('Says + statement'), remove for normalization.
        df_["statement"] = df_["statement"].str.replace(r"^(Says\s+|Says)", "", regex=True)

        # Replace nan
        df_ = self.replace_missing_values(df_)

        # Convert "subject" values to lists
        df_["subject"] = self.transform_string_to_list(df["subject"])

        # Lowercase
        df_ = df_.map(lambda x: x.lower() if isinstance(x, str) else x)

        return df_

    def deeper_processing(self, df):
        """This method applies a deeper processing, removing stop words, punctuation etc.
        It is modifying significantly the given full sentences like in column 'statement'. Therefore, it is applied
        after extracting POS.
        """
        df_ = df.copy()

        # subject is treated separately and party affiliation doesn't need extra processing
        for column in [
            "statement",
            "job_title",
            "state_info",
            "context",
        ]:
            df_[column] = df_[column].apply(
                lambda x: (
                    self.text_handler.preprocess_text(x, stop_words_basic=self.stop_words_basic)
                    if isinstance(x, str)
                    else x
                )
            )

        df_["speaker"] = df_["speaker"].apply(
            lambda x: self.text_handler.preprocess_text_simple(x) if isinstance(x, str) else x
        )

        # replace numbers with dd strings
        df_["statement"] = df_["statement"].apply(self.text_handler.replace_numbers)

        return df_

    def replace_missing_values(self, df):
        """Replace missing values with empty or specific string."""

        nan_replacement = self.nan_replacement_value

        fill_values = {col: nan_replacement for col in df.columns if col != "subject"}
        # subject treated separately

        df.fillna(fill_values, inplace=True)

        return df

    @staticmethod
    def transform_string_to_list(data: pd.Series):
        """Transform column "subject" to actual list of values and the nan to empty lists."""

        data_ = data.copy()

        data_ = data_.str.split(",")
        data_ = data_.apply(lambda x: [] if x is np.nan else x)  # if NaN values, replace with empty list

        return data_

    def concatenate_context_metadata(self, df):
        """Concatenates information from multiple columns into one column 'context_metadata'."""

        df_ = df.copy()

        cols_to_concat = ["speaker", "job_title", "party_affiliation", "state_info", "context"]

        if self.exclude_columns:
            cols_to_concat = [item for item in cols_to_concat if item not in self.exclude_columns]
            for col in self.exclude_columns:
                df_[col] = df_[col].fillna("unknown")  # not necessary now or is it?

        # some columns have missing data and 'affiliation' has str 'none'
        df_["rest_context"] = df_[cols_to_concat].apply(
            lambda row: " ".join(
                str(value) if all([pd.notna(value), value != "none"]) else "" for value in row
            ),
            axis=1,
        )

        # remove extra whitespaces and repeated words (also can still be completely empty)
        df_["rest_context"] = df_["rest_context"].apply(
            self.text_handler.remove_repetition_and_extra_whitespaces
        )

        df_.drop(columns=cols_to_concat, inplace=True)

        return df_

    def feature_extraction(self, dfx: pd.DataFrame) -> pd.DataFrame:
        """Extract linguistic features from the main text (statement).

        Specifically: num of tokens, average length of token, num of sentences and counts of some POS.
        It's applied on the 'statement' column.
        Currently not used / replaced by the Transformer.
        """
        text_handler = TextHandler()  # using self.text_handler throws error at grid-search/multiprocessing

        features = pd.DataFrame()
        features[["num_tokens", "avg_token_length"]] = dfx.apply(
            text_handler.count_tokens_and_avg_length
        ).apply(pd.Series)

        # Get POS counts as features
        pos_data = dfx.apply(text_handler.pos_counts)
        pos_features = pd.DataFrame(pos_data.tolist(), index=dfx.index)

        # Merge
        features_merged = pd.concat([features, pos_features], axis=1)
        self.feature_names = features_merged.columns.to_list()

        return features_merged

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Applies all the transformation explained in the class.

        Parameters:
        X : pandas DataFrame with the text columns

        Returns:
        X_ : pandas DataFrame with the data preprocessed and POS features extracted

        """
        df_ = X.copy()

        # Step 1: Some Basic preprocessing on the text but keeping it still intact for POS extraction
        df_processed = self.shallow_preprocessing(df_)

        # Step 2: Apply StatementFeatureExtractor to statement column
        new_features = self.feature_extractor.transform(df_processed["statement"])

        # Step 3: Now we can do more text processing, remove stop words etc.
        df_processed_ = self.deeper_processing(df_processed)

        X_ = pd.concat([df_processed_, new_features], axis=1)

        X_ = self.concatenate_context_metadata(X_)

        X_ = self.replace_missing_values(X_)

        self.final_columns = X_.columns.to_list()

        return X_

    def get_feature_names_out(self):
        # these are only the new feature names from POS extraction
        return self.feature_names


class StatementFeatureExtractor(BaseEstimator, TransformerMixin):
    """This Transformer extracts some linguistic features from the main text (statement).

    Specifically:
     - num of tokens
     - average length of token
     - num of sentences
     - some POS frequencies

    It's applied on the main 'statement' column.
    """

    def __init__(self):
        self.text_handler = TextHandler()
        self.feature_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        text_handler = TextHandler()  # needed in parallel processing like grid-search

        features = pd.DataFrame()
        features[["num_tokens", "avg_token_len"]] = X_.apply(text_handler.count_tokens_and_avg_length).apply(
            pd.Series
        )
        features["num_sentences"] = X_.apply(
            lambda x: len(text_handler.punkt_tokenizer.sentences_from_text(x))
        )

        # get POS counts as features
        pos_data = X_.apply(text_handler.pos_counts)
        pos_features = pd.DataFrame(pos_data.tolist(), index=X_.index)

        # Merge
        features_merged = pd.concat([features, pos_features], axis=1)
        self.feature_names = features_merged.columns.to_list()

        return features_merged

    def get_feature_names_out(self):
        return self.feature_names  # these are only the new feature names from POS extraction


class MultiLabelOneHotEncoder(BaseEstimator, TransformerMixin):
    """Custom One Hot encoder for multi label data."""

    def __init__(self):
        self.unique_labels_ = None

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters:
        X : pandas DataFrame with the data to fit.

        Returns:
        self : Returns the instance itself.
        """
        X_ = X.copy()

        unique_labels_ = set()
        for row in X_:
            if not isinstance(row, list):  # if nan
                row = []
            unique_labels_.update(row)

        # turn to sorted list
        self.unique_labels_ = sorted(unique_labels_)
        return self

    def transform(self, X):
        """
        Transform the data by one-hot encoding the multi-label items. Unseen at fitting labels will not be encoded.

        Parameters:
        X : pandas Series with the data to transform

        Returns:
        one_hot_encoded : pandas DataFrame with the one-hot encoded representation of the input data.
        """
        if self.unique_labels_ is None:
            raise ValueError("The encoder has not been fit yet.")

        X_ = X.copy()

        one_hot_encoded = []
        for row in X_:
            if not isinstance(row, list):
                row = []
            one_hot_row = [1 if label in row else 0 for label in self.unique_labels_]
            one_hot_encoded.append(one_hot_row)

        return pd.DataFrame(np.array(one_hot_encoded), columns=self.unique_labels_, index=X_.index)

    def get_feature_names_out(self):
        return self.unique_labels_


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """Custom One Hot Encoder (single label data)."""

    def __init__(self, prefix=""):
        self.unique_labels_ = None
        self.prefix = prefix

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters:
        X : pandas DataFrame with the data to fit.

        Returns:
        self : Returns the instance itself.
        """
        one_hot_df = pd.get_dummies(X, prefix=self.prefix)
        self.unique_labels_ = one_hot_df.columns.to_list()

        return self

    def transform(self, X):
        """
        Transform the data by one-hot encoding the multi-label items. Unseen at fitting labels will not be encoded.

        Parameters:
        X : pandas Series with the data to transform

        Returns:
        one_hot_encoded : pandas DataFrame with the one-hot encoded representation of the input data.
        """
        one_hot_encoded = []
        for row in X:
            one_hot_row = [1 if f"{self.prefix}_{row}" == label else 0 for label in self.unique_labels_]
            one_hot_encoded.append(one_hot_row)

        return pd.DataFrame(np.array(one_hot_encoded), columns=self.unique_labels_, index=X.index)

    def get_feature_names_out(self):
        return self.unique_labels_


class SentenceTransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X must be a list of sentences so transform it from pd.Series

        if not isinstance(X, list) and isinstance(X, pd.Series):
            X = X.tolist()
        else:
            raise ValueError("Input must be a Pandas Series or a list.")

        embeddings = self.model.encode(X)
        return embeddings


class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=400, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.text_handler = TextHandler()

    def fit(self, X, y=None):
        # Convert each document in X to a TaggedDocument
        tokenized_data = [self.text_handler.tokenize_text(sent) for sent in X]
        tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_data)]

        # Train the Doc2Vec model
        self.model = Doc2Vec(
            tagged_data,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )

        return self

    def transform(self, X):
        # Transform each document in X to its corresponding vector
        vectors = [self.model.infer_vector(self.text_handler.tokenize_text(doc)) for doc in X]
        vectors = np.array(vectors)
        return vectors
