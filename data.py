import torch
import torchvision
import torchvision.transforms as transforms


def build_dataset():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_test)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, testset


class DataSet:
    def __init__(self, split, batch_size):
        trainset, testset = build_dataset()
        if split == 0:
            self.train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                     num_workers=2)
        else:
            client_data_list = torch.utils.data.random_split(trainset, [len(trainset) // split for _ in range(split)])
            self.train = [torch.utils.data.DataLoader(clien_data, batch_size=batch_size, shuffle=True,
                                                      num_workers=2) for clien_data in client_data_list]

        self.test = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# dataset = DataSet(10, 64)