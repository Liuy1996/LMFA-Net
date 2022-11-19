import torch
import torch.nn as nn

class LMFANet(nn.Module):  #0.001
    def __init__(self, groups=16):
        super(LMFANet, self).__init__()
        self.groups = groups
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)

        self.conv3 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)

        self.box = nn.Sequential(
                nn.Conv2d(16,3, 5, padding=0, bias=True),
                nn.Sigmoid()
        )

    def Maxout(self, x, groups):
        x = x.reshape(x.shape[0], groups, x.shape[1]//groups, x.shape[2], x.shape[3])
        x, _ = torch.max(x, dim=2, keepdim=True)
        out = x.reshape(x.shape[0],-1, x.shape[3], x.shape[4])
        return out

    def forward(self, x):
        out0 = self.conv0(x)

        out1 = self.conv1(x)
        out1 = self.conv2(out1)

        out2 = self.conv3(x)
        out2 = self.conv4(out2)
        out2 = self.conv5(out2)

        y = torch.cat((out0,out1,out2), dim=1)
        y = self.Maxout(y, 16)
        out = self.box(y)

        return out

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total:', total_num, 'Trainable:' ,trainable_num)

if __name__ == "__main__":
    pass