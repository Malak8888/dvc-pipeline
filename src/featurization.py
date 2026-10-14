import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def get_df(path):
    return pd.read_csv(path, encoding="utf-8", header=None,
                       delimiter="\t", names=["id", "label", "text"])


def save_matrix(df, matrix, names, output):
    id_m = sparse.csr_matrix(df.id.astype(np.int64)).T
    lab_m = sparse.csr_matrix(df.label.astype(np.int64)).T
    result = sparse.hstack([id_m, lab_m, matrix], format="csr")
    with open(output, "wb") as fd:
        pickle.dump((result, names), fd)


def main():
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)["featurize"]
    if len(sys.argv) != 3:
        sys.exit("Usage: python featurization.py data-dir features-dir")

    in_path, out_path = sys.argv[1], sys.argv[2]
    os.makedirs(out_path, exist_ok=True)

    bow = CountVectorizer(stop_words="english",
                          max_features=params["max_features"],
                          ngram_range=(1, params["ngrams"]))
    tfidf = TfidfTransformer(smooth_idf=False)

    df_tr = get_df(os.path.join(in_path, "train.tsv"))
    words = np.array(df_tr.text.str.lower().values)
    bow.fit(words)
    names = bow.get_feature_names_out()
    counts = bow.transform(words)
    tfidf.fit(counts)
    save_matrix(df_tr, tfidf.transform(counts), names,
                os.path.join(out_path, "train.pkl"))

    df_te = get_df(os.path.join(in_path, "test.tsv"))
    words = np.array(df_te.text.str.lower().values)
    save_matrix(df_te, tfidf.transform(bow.transform(words)), names,
                os.path.join(out_path, "test.pkl"))


if __name__ == "__main__":
    main()