"""Some unit tests.
Run :
     pytest test_services.py
"""

import pandas as pd
import pytest

from services import TextHandler, MultiLabelOneHotEncoder

text_prep = TextHandler()


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "The profit margin is 25% while the company earned $1.2 million last year.",
            "The profit margin is 25 percent while the company earned $1 million last year.",
        ),
        (
            "40,000,500 people voted against the new law.",
            "40000500 people voted against the new law.",
        ),
    ],
)
def test_number_processing(text, expected):
    assert text_prep.preprocess_numbers(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "An estimated of 56 million Americans don't have bank accounts, not even 1.",
            "An estimated of dd million Americans don't have bank accounts, not even d.",
        ),
        (
            "400000 people voted against the new law. That is almost 50%.",
            "dddd people voted against the new law. That is almost dd%.",
        ),
    ],
)
def test_number_processing(text, expected):
    assert text_prep.replace_numbers(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "An ex-smoker, now non-smoker wrote an anti-smoking book.",
            "ex smoker now non smoker wrote anti smoking book",
        ),
        (
            "My 7-step plan creates 700,000 jobs in 7 years.",
            "My 7 step plan creates jobs in 7 years",
        ),
        (
            'The proposed education changes "will not cut one teacher\'s pay."',
            "proposed education changes will not cut one teacher pay",
        ),
    ],
)
def test_text_processing(text, expected):
    assert text_prep.preprocess_text(text) == expected


def test_multi_label_one_hot_encoder():
    train = pd.DataFrame({"subject": [["news", "sports"], [], ["media"]]})

    test = pd.DataFrame({"subject": [["news", "politics"], ["media", "finance"]]})

    encoder = MultiLabelOneHotEncoder()
    encoder.fit(train["subject"])

    test_encoded = encoder.transform(test["subject"])
    assert encoder.unique_labels_ == ["media", "news", "sports"]
    assert test_encoded.loc[0].to_list() == [0, 1, 0]
    assert test_encoded.loc[1].to_list() == [1, 0, 0]
