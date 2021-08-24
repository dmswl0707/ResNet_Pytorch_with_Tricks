# ResNet_Pytorch_with_Tricks

*Pytorch for ResNet with tricks.*

It is used with CIFAR10 / CIFAR100 data. And code ONLY for Usage




It contatined two ways.

☝🏻one thing is code for bringing the model from pytorch.

✌🏻other thing is code for making the model by self.

## Usage
> model.py  - code for making the model
> 
> preprocessing.py  - preprocess the dataset and execute data argumentation.
> 
> DataLoader.py  - download and load the dataset.
> 
> training.py  - train wetights.
> 
> main.py  - exucute training.py
> 
> Args.py  - Args for revise hyper parameters.

> inference  - the ways to test weights.
> 
> (topk accuracy.py, test by class.py, confusion matrix.py, CAM.py)
> 
> tricks  - the ways to add tricks, by training.
> 
> (cut-mix.py, early stopping.py, k-fold.py, WayToUse_Pretrained_ResNet.py)

## visualization
It also contain visualization tricks containing Class Activation Map, Cut-mix, Confusion Matrix code.
### Class Activation Map
<img src="https://user-images.githubusercontent.com/65028694/130565823-f4f49ea8-086f-4707-8162-709710ef1346.png"  width="680" height="470">

### Cut-mix
<img src="https://user-images.githubusercontent.com/65028694/130569484-1512e484-4f48-4893-8e2e-7f06d882a386.png"  width="680" height="470">

### Confusion Matrix
<img src="https://user-images.githubusercontent.com/65028694/130567874-5504eddb-8b5f-4bb7-b6eb-640f0239d2bf.png"  width="660" height="500">


## Network description 👇🏻👇🏻
https://www.notion.so/ResNet-94e2ee197aa24c9ea0a807953d26eee7

### Preview 
<img width="340" alt="스크린샷 2021-03-11 오전 1 21 47" src="https://user-images.githubusercontent.com/65028694/110661895-57d72980-8208-11eb-8ceb-6a6663fc7f6d.png">
<img width="502" alt="스크린샷 2021-03-11 오전 1 24 50" src="https://user-images.githubusercontent.com/65028694/110662215-a2f13c80-8208-11eb-85a1-87d440a3d73d.png">


Citation : He, Kaiming, et al. "Deep residual learning for image recognition. arXiv 2015." arXiv preprint arXiv:1512.03385 (2015).
