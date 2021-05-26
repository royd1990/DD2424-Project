from PIL import Image
from torchvision import transforms
from torchvision.datasets import STL10,MNIST
import cv2
from torchvision import torch
import numpy as np

np.random.seed(0)

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

class STL10Pair(STL10):
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        #img = torch.reshape(img,[1,img.shape[0],img.shape[1]])
       # img = img.view()
        print(img.shape)
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target

class MNISTPair(MNIST):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        #img = torch.reshape(img,[1,img.shape[0],img.shape[1]])
        img = Image.fromarray(np.array(img),'L')

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1 * 32)),
    transforms.ToTensor()])#,
    #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor()])#,
    #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
