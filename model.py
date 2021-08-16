from Args import *

# make model architecture
# resnet 50

act_fn = Args['Act_fu']

def conv_block_1(in_channel, out_channel, act_fn, stride = 1):
    model = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size = 1, stride = stride),
        act_fn
    )
    return model

def conv_block_3(in_channel, out_channel, act_fn, stride = 1):
    model = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = stride, padding = 1),
        act_fn
    )
    return model


class ResidualBlock(nn.Module): #Residual bottleneck
    def __init__(self, in_channel, mid_channel, out_channel, act_fn, down = False):
        super(ResidualBlock, self).__init__()
        self.act_fn = act_fn
        self.down = down #featur map이 줄어드는지 안 줄어드는지

        if self.down: #BottleNeck
            self.layer = nn.Sequential(
                conv_block_1(in_channel, mid_channel, act_fn, stride = 2),
                conv_block_3(mid_channel, mid_channel, act_fn),
                conv_block_1(mid_channel, out_channel, act_fn)
            )
            self.downsample = nn.Conv2d(in_channel, out_channel, kernel_size = 1, stride = 2) #stride = 2
        
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_channel, mid_channel, act_fn),
                conv_block_3(mid_channel, mid_channel, act_fn),
                conv_block_1(mid_channel, out_channel, act_fn)
            )
            self.dim_equalizer = nn.Conv2d(in_channel, out_channel, kernel_size = 1)

    def forward(self, x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else: 
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        return out


### model(ResNet)


class Model(nn.Module):
    def __init__(self, num_class):
        super(Model, self).__init__()
        self.act_fn = nn.ReLU()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.module1 = nn.Sequential(
            ResidualBlock(64, 64, 256, act_fn, down = False), #들어오고
            ResidualBlock(256, 64, 256, act_fn, down = False), # BT
            ResidualBlock(256, 64, 256, act_fn, down = True) #BT + down
            )
        self.module2 = nn.Sequential(
            ResidualBlock(256, 128, 512, act_fn,  down = False),
            ResidualBlock(512, 128, 512, act_fn, down = False),
            ResidualBlock(512, 128, 512, act_fn, down = False),
            ResidualBlock(512, 128, 512, act_fn, down = True)
            )
        self.module3 = nn.Sequential(
            ResidualBlock(512, 256, 1024, act_fn,  down = False),
            ResidualBlock(1024, 256, 1024, act_fn, down = False),
            ResidualBlock(1024, 256, 1024, act_fn, down = False),
            ResidualBlock(1024, 256, 1024, act_fn, down = False),
            ResidualBlock(1024, 256, 1024, act_fn, down = False),
            ResidualBlock(1024, 256, 1024, act_fn, down = True)
            )
        self.module4 = nn.Sequential(
            ResidualBlock(1024, 512, 2048, act_fn, down = False),
            ResidualBlock(2048, 512, 2048, act_fn, down = False),
            ResidualBlock(2048, 512, 2048, act_fn,  down = False)
            )
        self.avgpool = nn.AvgPool2d(1) #특징강화, 이미지 크기를 줄임
        self.fc = nn.Linear(2048, 10)

    def forward(self, input):
        x = input
        x = self.layer(x)

        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)

        x = self.avgpool(x)
        x = x.view(x.size()[0], -1) #Flatten layer
        x = self.fc(x) #classifier
        return x


model = Model(num_class=10)
print("Model")