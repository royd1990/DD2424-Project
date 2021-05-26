import argparse
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from tqdm import tqdm
import utils
from network import Network


def train_test(net, data_loader, training_optimizer):
    is_train = training_optimizer is not None

    if is_train:
        net.train()
    else:
        net.eval()

    total_loss = 0
    total_correct_1 = 0
    total_correct_5 = 0
    total_num = 0
    data_progress = tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_progress:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                training_optimizer.zero_grad()
                loss.backward()
                training_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_progress.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                          .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results/model_400.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')

    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    train_data = STL10(root='data', split='train', transform=utils.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_data = STL10(root='data', split='test', transform=utils.test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = Network(num_class=len(train_data.classes), pretrained_path=model_path).cuda()
    for param in model.layer.parameters():
        param.requires_grad = False
    model = nn.DataParallel(model)

    adam = optim.Adam(model.module.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_test(model, train_loader, adam)
        test_loss, test_acc_1, test_acc_5 = train_test(model, test_loader, None)
