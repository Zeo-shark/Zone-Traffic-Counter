import torch.nn as nn
import torch
from torchvision import models
import collections

class CANNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CANNet,self).__init__()
        self.frontend_feat=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat=[512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 1024,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.conv1_1=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv1_2=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv2_1=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv2_2=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv3_1=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv3_2=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv6_1=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv6_2=nn.Conv2d(512,512,kernel_size=1,bias=False)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
#
            fsd=collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key=list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key]=list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)