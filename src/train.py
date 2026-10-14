import os
import pickle
import sys

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier


def main():
    with open("params.yaml", encoding="utf-8") as f:
        p = yaml.safe_load(f)["train"]
    if len(sys.argv) != 3:
        sys.exit("Usage: python train.py features-dir model.pkl")

    with open(os.path.join(sys.argv[1], "train.pkl"), "rb") as fd:
        matrix, _ = pickle.load(fd)

    y = np.squeeze(matrix[:, 1].toarray())
    X = matrix[:, 2:]

    clf = RandomForestClassifier(n_estimators=p["n_est"],
                                 min_samples_split=p["min_split"],
                                 n_jobs=2, random_state=p["seed"])
    clf.fit(X, y)

    with open(sys.argv[2], "wb") as fd:
        pickle.dump(clf, fd)


if __name__ == "__main__":
    main()