from sklearn.model_selection import StratifiedKFold, KFold


def stratified_kfold(ids, stratify, num_splits, random_state=42):
    kfold = StratifiedKFold(num_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kfold.split(ids, stratify):
        yield [ids[i] for i in train_idx], [ids[i] for i in test_idx]


def kfold(ids, num_splits, random_state=42):
    kfold = KFold(num_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kfold.split(ids):
        yield [ids[i] for i in train_idx], [ids[i] for i in test_idx]
