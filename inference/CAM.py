import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
from training import model
from torch.autograd import Variable
import torch.nn as nn
from DataLoader import testloader
from Args import *

#print(model)
conv = list(model.children())[-3]
#print(conv)
params = list(model.parameters())
features = params[-3].size()
#print(features)


PATH='/home/eunji/project_dir/cifar+ResNet/custom_model.pt'
model.load_state_dict(torch.load(PATH))

device = Args["device"]
model = model.to(device)

class CAM():
    def __init__(self, model):
        self.gradient = []
        self.h = list(model.children())[-3].register_backward_hook(self.save_gradient)

    def save_gradient(self, *args):
        grad_input = args[1]
        grad_output = args[2]
        self.gradient.append(grad_output[0])

    def get_gradient(self, idx):
        return self.gradient[idx]

    def remove_hook(self):
        self.h.remove()

    def normalize_cam(self, x):
        x = 2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8) - 1
        x[x < torch.max(x)] = -1
        return x

    def visualize(self, cam_img, img_var):
        cam_img = resize(cam_img.cpu().data.numpy(), output_shape=(32, 32))
        x = img_var[0, :, :].cpu().data.numpy()

        plt.subplot(1, 3, 1)
        plt.imshow(cam_img)

        plt.subplot(1, 3, 2)
        plt.imshow(x, cmap="gray")

        plt.subplot(1, 3, 3)
        plt.imshow(x + cam_img)
        plt.show()

    def get_cam(self, idx):
        grad = self.get_gradient(idx)
        alpha = torch.sum(grad, dim=3, keepdim=True)
        alpha = torch.sum(alpha, dim=2, keepdim=True)

        cam = alpha[j] * grad[j]
        cam = torch.sum(cam, dim=0)
        cam = self.normalize_cam(cam)

        self.remove_hook()
        return cam

cam = CAM(model)

for i, [image, label] in enumerate(testloader):
    x = Variable(image).cuda()
    y_ = Variable(label).cuda()

    output = model.forward(x)

    for j in range(20):
        model.zero_grad()
        lab = y_[j].cuda().item() #cpu()
        output[j, lab].backward(retain_graph=True)

        out = cam.get_cam(j)
        cam.visualize(out, x[j])

    break



