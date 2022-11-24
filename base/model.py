from torchsummary import summary
from torch import nn
import torchvision.transforms as T


def conv(inp, out, act=True):
    activation = nn.PReLU() if act else nn.Identity()
    layer = nn.Sequential(
        nn.Conv2d(inp, out, 3, padding='same'),
        nn.BatchNorm2d(out),
        activation
        )
    return layer

class block(nn.Module):
    def __init__(self,
                 inp, out,
                 layers=2,
                 act=nn.PReLU()):
        super().__init__()
        self.layers = nn.Sequential(
            conv(inp, out),
            *[conv(out, out) for i in range(layers-2)],
            conv(out, out, False)
            )
        self.shortcut = nn.Sequential(
            nn.Conv2d(inp, out, 1),
            nn.BatchNorm2d(out)
            ) if inp != out else nn.Identity()
        self.act = act

    def forward(self, x):
        res = self.shortcut(x)
        x = self.layers(x)
        x += res
        return self.act(x)

class BaseNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Sequential(
                 T.RandomHorizontalFlip(p=0.2),
                 T.RandomRotation(degrees=(0, 30)),
                 )  # искусственно увеличиваем данные

        self.encod = nn.Sequential(
            block(3, 6, 3),  #(3, 224, 224)
            nn.MaxPool2d(2), #-> (6, 112, 112)
            block(6, 12, 3), 
            nn.MaxPool2d(2), #-> (12, 56, 56)
            block(12, 24, 3),
            nn.MaxPool2d(2), #-> (24, 28, 28)
            block(24, 48, 3),
            nn.MaxPool2d(2), #-> (48, 14, 14)
            block(48, 48, 3),
            nn.MaxPool2d(2), #-> (48, 7, 7)
            nn.AdaptiveAvgPool2d((5, 5)), #-> (48, 5, 5)
            nn.Flatten() #-> (1200)
            )
        self.fc = nn.Sequential(
                 nn.Linear(1200, 200),
                 nn.Tanh(),

                 nn.Linear(200, 50),
                 nn.Tanh(), 

                 nn.Linear(50, 5),
                 nn.BatchNorm1d(5),
                 nn.Softmax(-1)     
                 )

    def forward(self, x, trans=True):
        if trans:
            x = self.transform(x)

        x = self.encod(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    net = BaseNN()
    summary(net, (3, 224, 224))
