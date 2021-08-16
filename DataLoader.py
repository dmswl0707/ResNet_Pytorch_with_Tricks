import torchvision
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from Args import *
from preprocessing import *
import numpy as np
import matplotlib.pyplot as plt


# Data_loader
# 데이터 불러오기 및  데이터 로드


# cifar 10과 cifar 100 불러오기

trainset = torchvision.datasets.CIFAR100(root='./data',train = True, download = True, transform = transforms_train)
testset = torchvision.datasets.CIFAR100(root='./data',train = False, download = True, transform = transforms_test)

#trainset = torchvision.datasets.CIFAR10(root='./data',train = True, download = True, transform = transforms_train)
#testset = torchvision.datasets.CIFAR10(root='./data',train = False, download = True, transform = transforms_test)

num_train = len(trainset)
num_test = len(testset)
val_size = 0.2
#print({'train' : num_train})
#print({'test' : num_test})

indice = list(range(num_train))
np.random.shuffle(indice)

split = int(np.floor(val_size*num_train))
val_idx, train_idx = indice[:split], indice[split:]
#print(len(val_idx), len(train_idx))

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)


batch_size = Args["batch_size"]

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler = train_sampler, num_workers=6)
valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler = val_sampler, num_workers=6)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=6)

categories = list(trainset.class_to_idx.keys())
#print(categories)
num_class = len(categories)
#print(num_class)

'''
### 이미지 확인
def im_convert(tensor):
  image = tensor.clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array([0.5, 0.5, 0.5] + np.array([0.5, 0.5, 0.5]))
  image = image.clip(0, 1)
  return image


dataiter = iter(trainloader)
images, labels = dataiter.next()

fig = plt.figure(figsize=(25, 4))

for i in np.arange(20):
  # row 2 column 10
  ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[i]))
  ax.set_title(categories[labels[i].item()])
  #plt.show()
'''