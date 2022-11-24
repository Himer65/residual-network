import numpy as np
from train import *
from base.model import *

torch.manual_seed(123)
device = 'cuda'

parameters = torch.load('/content/drive/MyDrive/base.pt', map_location=device)

base = BaseNN().to(device)
base.load_state_dict(parameters[0])
hist = parameters[2]

dataset = np.load('/content/drive/MyDrive/anime-dataset.npz')
dataset = list(zip(torch.from_numpy(dataset['x'].reshape(-1, 3, 224, 224)).to(device),
                   torch.from_numpy(dataset['y']).to(device)))

train_loader = torch.utils.data.DataLoader(dataset[:-2500],
                                           batch_size=100,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset[-2500:],
                                           batch_size=1,
                                           shuffle=False)

optimizer = torch.optim.Adam(base.parameters(),
                             lr=1e-3,
                             weight_decay=1e-4)
optimizer.load_state_dict(parameters[1])

loss_func = nn.CrossEntropyLoss()

histNew = training(model=base,
                   train_loader=train_loader,
                   val_loader=valid_loader, 
                   opt=optimizer,
                   loss_func=loss_func,
                   steps=2)

hist['l-val'] += histNew['l-val']
hist['l-train'] += histNew['l-train']

torch.save([base.state_dict(),
            optimizer.state_dict(),
            hist], '/content/drive/MyDrive/base.pt')
visualiz(hist)
print(f'accuracy: {accuracy(base, valid_loader)}')
