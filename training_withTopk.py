from DataLoader import *
from functions.AverageMeter import *
#from model import *
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

model = models.resnet50(pretrained = False)

if torch.cuda.device_count()>1:
    net = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

model = net.to(Args["device"])
#print(Model)

Epoch = Args["Epoch"]
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), Args["lr"], momentum=0.9, weight_decay=1e-4)

save_path = '/workspace/pytorch/project_dir/Ratio_Image_Recognition/weights/' + 'model_state_dict.pt'


def train():

    for epoch in range(Epoch):
        print("===================================================")
        print("epoch: ", epoch + 1)

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        model.train()
        '''
        if Epoch > 1 :
            Model.load_state_dict(torch.load(save_path))
            Model.to(Args["device"])
        '''
        for i, (images, target) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(Args["device"])
            target = target.to(Args["device"])
            #print("Outside: input size", images.size(),"output_size", target.size())

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            loss.backward()
            optimizer.step()

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        torch.save(model.module.state_dict(), save_path)


def validate():

    for epoch in range(Epoch):
        print("===================================================")
        print("epoch: ", epoch + 1)

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        model.eval()

        with torch.no_grad():

            for i, (images, target) in enumerate(val_loader):
                images = images.cuda()
                target = target.cuda()

                output = model(images)
                loss = criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1,5))

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))


train()