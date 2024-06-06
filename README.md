# Fact or Act?

### Welcome to my project

It is my humble attempt to contribute in the detection and elimination of fake information while being creative, having fun, learning and expanding or brushing up my ML, NLP and coding skills. 👩🏻‍💻

This project is using the [LIAR Dataset](https://aclanthology.org/P17-2067/) and the goal is to build a classifier able to distinguish between facts and lies. Several techniques and algorithms are being explored in an effort to find the best combination of text preprocessing, feature engineering and vectorization techniques as well as classification algorithms.



### More info:

**Feature Engineering**:
The idea so far is to treat different columns differently, with `statement` being the main text feature. 
- The `statement` column is the one that is more experimented with, as this the core part in question of validity. So different vectorizations have been explored and more to come.
- For low cardinality columns like `subject` and `party_affiliation` so far one-hot encoding is applied, as both have been found to be important for the target.
- The rest columns for now, are concatenated in a secondary '_rest context_' part of data and for now mostly vectorized with TFIDF.
- Finally, and very importantly, some linguistic features are extracted from the `statement` and used as well in the feature space.

Some of the text vectorization techniques that have been explored so far:
- TFIDF
- [Doc2Vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html)
- [Sentence Transformers](https://www.sbert.net/)

Last, for now the problem has been turned into a **binary** classification.

Information about certain choices made can be found in my  [Implementation Decision Records file](IDRs.md).

Disclaimer*:  
_While main functionality is achieved, this project remains **work in progress** in the sense that new approaches will still be explored and also the code is not yet polished or very efficient._ 


### Contents
The code contains the necessary functionality as well as the functionality of components that were implemented as part of the exploration and experimentation.
Following files are included:

- `main.py`: The script contains a pipeline of the main functionality, building and saving the model, testing the performance on a test subset and also predicting a random sample
- `services.py`: Includes objects that apply transformations on the data, such as transformers
- `utils.py`: Some generic utility methods
- `test_services`: A few unit tests
- `IDRs.md`: an Implementation Decision Records file, explaining and documenting the different choices made and explored

### **Upcoming Improvements:**

Soon to come and/or continuously explored:
- profiling and code inefficiencies, particularly in preprocessing step
- data exploration (data was explored, just not polished yet)
- benchmarking
- possibly a nice UI with Streamlit
- exploration of other techniques, more linguistic features, text preprocessing
- feature selection and more.

###

------------------------------------------------------------------------------------
## Installation

To install this project and the dependencies you can use anaconda:


```bash
   cd <project dir> 

   conda create -n <env name> python==3.10.0 pip==23.3.2

   conda activate <env name>

   pip install -r requirements.txt
```



To build a model and get a prediction on a random sample of the test data, run the main method:

```bash
  python main.py
```
In `main.py`, a model will be trained using a subset of the train data and the pipeline that contains both the processing and the classifier will be saved as
`.joblib` file, to be used later for inference.

###

---

## Data & dataset

This project is using the [LIAR Dataset](https://aclanthology.org/P17-2067/). The dataset is not pushed to github.
You can download it from the [official_url](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) in .tsv files. Alternatively you can get it through [Hugging Face](https://huggingface.co/datasets) using the [dataset](https://huggingface.co/docs/datasets/v1.5.0/loading_datasets.html#) library.

The original dataset was slightly processed (e.g. replacing json.ids as id and later index) and the total credit history counts columns have been removed.
Furthermore, for the time being, labels were turned to a binary classification (possibly to change).

The data now looks like this: 

| label  | statement                                   | subject      | speaker        | job_title              | state_info   | party_affiliation | context                          |
|--------|---------------------------------------------|--------------|----------------|------------------------|--------------|-------------------|----------------------------------|
| FALSE  | I'm pro-life. He's not.                     | abortion     | sam-brownback  | U.S. senator           | Kansas       | republican        | nan                              |
| TRUE   | Says New Jersey is 50th in return of our .. | states,taxes | chris-christie | Governor of New Jersey | New Jersey   | republican        | an interview on NBC Nightly News |

I will eventually push also code for the dataset file processing.

##

----------
## Troubleshooting

1. Error: `OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.`

    You should download spacy's language package from the terminal like this: 
    ```bash
    python -m spacy download en_core_web_sm
    ```
