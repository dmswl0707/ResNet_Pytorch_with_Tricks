import sys
sys.path.insert(0, '/home/eunji/early-stopping-pytorch')

import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler
import torchvision.models as models
from pytorchtools import EarlyStopping
from DataLoader import trainloader,valloader
from Args import *
import wandb
from model_f import Model
#from model import *


#model = models.resnet50(pretrained=False)
model = Model(num_classes=100)


# wandb 오프라인으로 돌리기

#os.environ["WANDB_API_KEY"] = 'ec46f0bd08d3fa6649b39fe9096ee68553dbf66b'
#os.environ["WANDB_MODE"] = "dryrun"

wandb.init()

# train setting
criterion = nn.CrossEntropyLoss()
device = Args["device"]

network = model.to(device)
print(device)

optimizer = optim.Adam(model.parameters(), lr=Args["lr"], betas=(0.9, 0.999), eps=1e-08, weight_decay=Args["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=Args["eta_min"])

Epoch = Args["Epoch"]
patience = Args["patience"]

wandb.watch(network)

def training():

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(Epoch):
        print("===================================================")
        print("epoch: ", epoch + 1)

        train_loss, val_loss = [], []
        avg_train_loss, avg_val_loss =  [], []

        total = 0
        v_total = 0

        train_loss = 0.0
        train_correct = 0.0
        val_loss = 0.0
        val_correct = 0.0

        model.train()
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            X = inputs.to(device)
            y = labels.to(device)
            '''
            # cut mix data argumentation
            
            if Args["beta"] > 0 and np.random.random() > 0.5:  # cutmix 작동될 확률
                lam = np.random.beta(Args["beta"], Args["beta"])
                rand_index = torch.randperm(X.size()[0]).to(Args["device"])
                target_a = y
                target_b = y[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
                X[:, :, bbx1:bbx2, bby1:bby2] = X[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2]))
                y_pred = network(X)
                loss = criterion(y_pred, target_a) * lam + criterion(y_pred, target_b) * (1. - lam)
            '''
            #else:
            y_pred = network(X)
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()

            # acc를 구하는 부분
            _, preds = torch.max(y_pred, 1)

            total += y.size(0)
            train_loss += loss.item()
            #train_loss.append(loss.item())
            train_correct += (preds == y).sum().item()

            epoch_loss = train_loss / len(trainloader)
            epoch_acc = 100. * float(train_correct) / total

        #train_loss_history.append(epoch_loss)
        #train_correct_history.append(epoch_acc)

        print("train loss: {:.4f}, acc: {:4f}".format(epoch_loss, epoch_acc))
        #scheduler.step()

        wandb.log({
            "Train Loss": epoch_loss,
            "Train Accuracy": epoch_acc,
            "Train error": 100 - epoch_acc,
            "lr" : optimizer.param_groups[0]['lr'] # 학습률 로깅

        })

        with torch.no_grad():

            model.eval()
            for val_inputs, val_labels in valloader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                y_val_pred = network(val_inputs)
                v_loss = criterion(y_val_pred, val_labels)

                #validation - acc를 구하는 부분
                _, v_preds = torch.max(y_val_pred, 1)

                v_total += val_labels.size(0)
                val_loss += v_loss.item()
                #val_loss.append(v_loss.item())
                val_correct += (v_preds == val_labels).sum().item()

                val_epoch_loss = val_loss / len(valloader)
                val_epoch_acc = 100. * float(val_correct) / v_total


            np_train_loss = np.average(train_loss)
            np_val_loss = np.average(val_loss)
            avg_train_loss.append(np_train_loss)
            avg_val_loss.append(np_val_loss)

            #val_loss_history.append(val_epoch_loss)
            #val_correct_history.append(val_epoch_acc)

            print("val loss: {:.4f}, acc: {:4f}".format(val_epoch_loss, val_epoch_acc))
            scheduler.step()

            wandb.log({
                "Val Loss": val_epoch_loss,
                "Val Accuracy": val_epoch_acc,
                "Val error": 100 - val_epoch_acc,
            })

            early_stopping(val_epoch_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break


