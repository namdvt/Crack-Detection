import torch
import torch.optim as optim
from helper import write_log, write_figures
import numpy as np
from dataset import get_loader
import torch.nn.functional as F

from model import Model
from tqdm import tqdm


def fit(epoch, model, optimizer, device, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0.0

    for image, label in tqdm(data_loader):
        image = image.to(device)
        label = label.to(device)

        if phase == 'training':
            optimizer.zero_grad()
            output = model(image)
        else:
            with torch.no_grad():
                output = model(image)

        # loss
        loss = F.cross_entropy(output, label)
        running_loss += loss.item()

        # acc
        predict = torch.argmax(output, dim=1)
        running_correct += (predict == label).sum().item()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(data_loader)
    epoch_acc = running_correct / len(data_loader.dataset)

    print('[%d][%s] loss: %.4f acc %.4f' % (epoch, phase, epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc


def train():
    print('start training ...........')
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = Model().to(device)

    batch_size = 32
    num_epochs = 500
    learning_rate = 0.005
    train_loader, val_loader = get_loader(root='data/crack-identification-ce784a-2020-iitk/train', batch_size=2)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.1)

    train_losses, val_losses = [], []
    train_acc, val_acc = [], []
    num_not_improve = 0
    for epoch in range(num_epochs):
        train_epoch_loss, train_epoch_acc = fit(epoch, model, optimizer, device, train_loader, phase='training')
        val_epoch_loss, val_epoch_acc = fit(epoch, model, optimizer, device, val_loader, phase='validation')
        print('-----------------------------------------')

        if epoch == 0 or val_epoch_acc >= np.max(val_losses):
            torch.save(model.state_dict(), 'output/weight.pth')
            num_not_improve = 0
        else:
            num_not_improve += 1

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        train_acc.append(train_epoch_acc)
        val_acc.append(val_epoch_acc)

        write_figures('output', train_losses, val_losses, train_acc, val_acc)
        write_log('output', epoch, train_epoch_loss, val_epoch_loss, train_epoch_acc, val_epoch_acc)
        scheduler.step()

        if num_not_improve == 20:
            print('early stop')
            break


if __name__ == "__main__":
    train()