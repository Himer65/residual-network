import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

@torch.no_grad()
def testing(model,
            loader,
            loss_func):
    model.eval()
    l = 0

    for batch, (x, y) in enumerate(loader, 1):
        pred = model(x/255, False)
        loss = loss_func(pred, y.float())
        l += loss.item()
  
    return l/batch

def training(model, 
             train_loader,
             val_loader,
             opt,
             loss_func,
             steps) -> dict:

    hist = {'l-val': [],
            'l-train': []}

    for st in range(1, steps+1):
        model.train()
        lt = 0

        for batch, (x, y) in enumerate(train_loader, 1):
            pred = model(x/255)
            loss = loss_func(pred, y.float())
            loss.backward()
            opt.step()
            opt.zero_grad()

            lt = (lt + loss.item())/2
            if batch%50 == 0:
                print(f'epoch: {st}|t-loss: {lt:.3f} <{batch}/{len(train_loader)}>')

        lv = testing(model,
                     val_loader, 
                     loss_func)
        print(f'epoch: {st}|v-loss: {lv:.3f} <{batch}/{len(train_loader)}>\n')
        hist['l-val'].append(lv)
        hist['l-train'].append(lt)

    return hist

@torch.no_grad()
def accuracy(model, data):
    model.eval()
    acc = 0

    for x, y in data:
        pred = model(x/255)
        acc += accuracy_score(y.cpu(), torch.round(pred).cpu())

    return acc/len(data)

def visualiz(hist: dict):
    for key in hist:
        l = range(len(hist[key]))
        plt.plot(l, hist[key], label=str(key))
    
    plt.legend(loc='lower left')
    plt.savefig('history.png')
    plt.show()
