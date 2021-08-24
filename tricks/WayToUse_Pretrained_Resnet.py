import torchvision.models as models
import torch.nn as nn
from Args import *

# pretrained model
# imageNet으로 학습된 모델 가져와서 레이어 변형하기

# 1. fc 층만 바꾸기
# 2. 레이어 추가하기(imagenet을 학습할 만한 리소스가 있어야함)

model_ft = models.resnet152(pretrained=True)
model = model_ft
#print(model_ft)

'''
# 특징 추출기 부분 선택하기
self.features = nn.Sequential(*(list(model.children())[0:8]))

# 특징추출기의 파라미터 픽스하기
for param in self.features.parameters():
    param.requires_grad = False
    
-> requires_grad = False가 weights를 얼리는 것
'''

# 1. fc 층 바꾸기

features = nn.Sequential(*list(model_ft.children())[0:7])
end = list(model.children())[-1]
print(end)

for param in features.parameters():
    param.requires_grad = False
'''
# fc in_feature 고정하기
num_in = model_ft.fc.in_features
# 새로운 데이터셋 클래스 갯수
num_out = 10

#new_fc
model_ft.fc = nn.Linear(in_features=num_in, out_features=num_out)

model = model_ft.to(Args["device"])

## 학습 루프 실행 ...
#training()
'''

# 2. 레이어 추가 하기

# 추가해 줄 레이어
'''
class SELayer(nn.Module):
    def __init__(self, in_channeld, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channeld, in_channeld // reduction, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(in_channeld // reduction, in_channeld, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

model_ft.avgpool = nn.Sequential(model_ft.avgpool, SELayer(in_channeld=2048, reduction=16))

model = model_ft.to(Args["device"])
print(model)

# avgpool 레이어 전까지 가중치를 freeze 해줌
# model_ft.avgpool 뒤에 SElayer가 들어감
# model_ft의 avgpool 레이어부터 끝단 레이어까지 새로 학습이 필요함

## 학습 루프 실행 ...(imageNet으로 학습이 필요함)
#training()
'''

