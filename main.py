import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
from model import Model


def train(net, data_loader, train_optimizer, temperature, loss_type, tau_plus):
    net.train()
    total_loss = 0
    total_count = 0
    train_progress = tqdm(data_loader)
    for pos_1, pos_2, target in train_progress:
        pos_1 = pos_1.cuda(non_blocking=True)
        pos_2 = pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = utils.get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if loss_type:
            N = batch_size * 2 - 2
            Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        else:
            Ng = neg.sum(dim=-1)

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_count += batch_size
        total_loss += loss.item() * batch_size

        current_loss = total_loss / total_count
        train_progress.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, current_loss))

    training_loss = total_loss / total_count
    return training_loss


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(network, memory_data_loader, test_set_loader):
    network.eval()
    total_top1 = 0
    total_top5 = 0
    total_num = 0
    feature_bank = []

    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Extract Features'):
            feature, out = network(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_targets = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_progress = tqdm(test_set_loader)
        for data, _, target in test_progress:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = network(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank -> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)

            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_targets.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score -> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_progress.set_description('KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                          .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--input_dim', default=128, type=int, help='Input dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used in prediction')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images in a mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the train set')
    parser.add_argument('--loss_type', default=True, type=bool, help='Debiased contrastive loss or standard loss')

    # args parse
    args = parser.parse_args()
    input_dim = args.input_dim
    temperature = args.temperature
    tau_plus = args.tau_plus
    k = args.k
    batch_size = args.batch_size
    epochs = args.epochs
    loss_type = args.loss_type

    # data prepare
    #train_data = utils.STL10Pair(root='data', split='train+unlabeled', transform=utils.train_transform)
    train_data = utils.MNISTPair(root='data', train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True,
                              drop_last=True)
    memory_data = utils.MNISTPair(root='data', train=True, transform=utils.test_transform)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_data = utils.MNISTPair(root='data', train=False, transform=utils.test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # model setup and optimizer config
    model = Model(input_dim).cuda()
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)
    print('# Classes: {}'.format(c))

    # training loop
    if not os.path.exists('results'):
        os.mkdir('results')
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, temperature, loss_type, tau_plus)
        if epoch % 25 == 0:
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
            torch.save(model.state_dict(), 'results/model_{}.pth'.format(epoch))
