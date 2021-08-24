import torchvision.transforms as transforms
from Args import *


# 데이터 전처리 및 증강

transforms_train = transforms.Compose([transforms.Resize((32, 32)),
                                 #transforms.RandomCrop(32, padding=4),
                                 #transforms.RandomRotation(degrees=30),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(Args["mean"], Args["std"])
                                ])

transforms_test = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(Args["mean"], Args["std"])
                                      ])






