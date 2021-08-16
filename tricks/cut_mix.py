import numpy as np
import torch
import matplotlib.pyplot as plt
from Args import *
from DataLoader import trainloader

# cut mix Data Argumentaion
# 랜덤하게 패치를 만들어 내어 다른 이미지에 합성하는 Data argumentaion 기법


def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] # 이미지의 width
    H = size[3] # 이미지의 height
    cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기
    cut_w = np.int(W * cut_rat)  # 패치의 너비
    cut_h = np.int(H * cut_rat)  # 패치의 높이

    # uniform
    # 기존 이미지의 크기에서 랜덤하게 값을 가져옵니다.(중간 좌표 추출)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출합니다.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# 배치 내의 데이터 셔플

X,y = next(iter(trainloader))
X = X.to(Args["device"])
y = y.to(Args["device"])

lam = np.random.beta(1.0, 1.0)  # 베타 분포에서 lam 값을 가져옵나다.
rand_index = torch.randperm(X.size()[0]).to(Args["device"]) # batch_size 내의 인덱스가 랜덤하게 셔플됩니다.
shuffled_y = y[rand_index] # 타겟 레이블을 랜덤하게 셔플합니다.

#print(lam)
#print(rand_index)

# 패치 부분 교체하기

bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
X[:,:,bbx1:bbx2, bby1:bby2] = X[rand_index,:,bbx1:bbx2, bby1:bby2]

# 람다 조정하기

lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2]))
#print(lam)

# 결과 확인하기

plt.imshow(X[0].permute(1, 2, 0).cpu())
#plt.show()


#  training에 적용

def cutmix_plot(trainloader):
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(15, 12)

    for i in range(3):
        for inputs, targets in trainloader:
            inputs = inputs
            targets = targets
            break

        lam = np.random.beta(1.0, 1.0)
        rand_index = torch.randperm(inputs.size()[0])
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
        axes[i].imshow(inputs[1].permute(1, 2, 0).cpu())
        axes[i].set_title(f'λ : {np.round(lam, 3)}')
        axes[i].axis('off')
    #plt.show()
    return


#cutmix_plot(trainloader)
