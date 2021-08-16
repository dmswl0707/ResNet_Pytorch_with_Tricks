from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from DataLoader import trainset, train_sampler, val_sampler
from training import training, model, criterion, optimizer
from Args import *
import torch

# K fold validation
# 데이터가 적거나, 평가의 신뢰성을 높일 때 사용합니다.


def kfold_Dataloader_set(train_df, valid_df=None):

    if valid_df is not None:  # k-fold
        train_set = train_sampler

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=Args["batch_size"], sampler = train_sampler, shuffle=True, drop_last=True, num_workers=1)
        valid_set = val_sampler
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, sampler = val_sampler, shuffle=False, num_workers=1)
    else:
        train_set, valid_set = train_test_split(trainset, test_size=0.1, random_state=42, stratify=train_df["invasive"])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=Args["batch_size"], shuffle=True, drop_last=True, num_workers=1)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, valid_loader


def fold_train(Args, train_df):

    folds = StratifiedKFold(n_splits=Args["NUM_FOLDS"], shuffle=True, random_state=42)

    X = train_df

    for i, (train_index, valid_index) in enumerate(folds.split(X,X["invasive"])):
        fold_num = i+1
        X_train = X.iloc[train_index]
        X_val = X.iloc[valid_index]

        Model = model
        Criterion = criterion
        Optimizer = optimizer


        train_loader, valid_loader = kfold_Dataloader_set(X_train, X_val)

        print("=" * 100)
        print(f"{fold_num}/{Args['num_fold']} Cross Validation Training Starts ...\n")

        #training()

        print(f"\n{fold_num}/{Args['num_fold']} Cross Validation Training Ends ...\n")

    return
