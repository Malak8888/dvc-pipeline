import os
import random
import re
import sys
import xml.etree.ElementTree

import yaml


def process_posts(lines, fd_train, fd_test, target_tag, split):
    for num, line in enumerate(lines, 1):
        try:
            fd_out = fd_train if random.random() > split else fd_test
            attr = xml.etree.ElementTree.fromstring(line).attrib
            pid = attr.get("Id", "")
            label = 1 if target_tag in attr.get("Tags", "") else 0
            title = re.sub(r"\s+", " ", attr.get("Title", "")).strip()
            body = re.sub(r"\s+", " ", attr.get("Body", "")).strip()
            fd_out.write(f"{pid}\t{label}\t{title} {body}\n")
        except Exception as ex:
            sys.stderr.write(f"Skipping broken line {num}: {ex}\n")


def main():
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)["prepare"]
    if len(sys.argv) != 2:
        sys.exit("Usage: python prepare.py data-file")

    random.seed(params["seed"])
    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

    with open(sys.argv[1], encoding="utf-8") as f:
        lines = f.readlines()

    with open(os.path.join("data", "prepared", "train.tsv"), "w", encoding="utf-8") as tr, \
         open(os.path.join("data", "prepared", "test.tsv"), "w", encoding="utf-8") as te:
        process_posts(lines, tr, te, "<r>", params["split"])


if __name__ == "__main__":
    main()