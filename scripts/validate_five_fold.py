#!/usr/bin/env python3
"""Validate the fixed five-fold assignment without creating a new split."""

import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def read_ids(path):
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main():
    primary = REPO / "data/primary_benchmark"
    path = primary / "five_fold/assignment.tsv"
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    ids = [row["public_id"] for row in rows]
    assigned = set(ids)
    train_ids = read_ids(primary / "train.txt")
    val_ids = read_ids(primary / "val.txt")
    test_ids = read_ids(primary / "test.txt")

    assert len(train_ids) == 1451 and len(val_ids) == 139 and len(test_ids) == 110
    assert not train_ids.intersection(val_ids)
    assert not train_ids.intersection(test_ids)
    assert not val_ids.intersection(test_ids)
    assert len(rows) == 1590 and len(assigned) == 1590
    assert assigned == train_ids.union(val_ids)
    assert assigned.isdisjoint(test_ids)

    folds = {int(row["validation_fold"]) for row in rows}
    assert folds == {1, 2, 3, 4, 5}
    validation_occurrences = {public_id: 0 for public_id in assigned}
    for fold in sorted(folds):
        fold_val = {row["public_id"] for row in rows if int(row["validation_fold"]) == fold}
        fold_train = assigned - fold_val
        assert len(fold_train) == 1272 and len(fold_val) == 318
        assert not fold_train.intersection(fold_val)
        for public_id in fold_val:
            validation_occurrences[public_id] += 1
        print(f"fold_{fold}: train={len(fold_train)} val={len(fold_val)}")

    assert set(validation_occurrences.values()) == {1}
    print("five_fold_membership=PASS")


if __name__ == "__main__":
    main()
