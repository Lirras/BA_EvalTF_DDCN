import torchvision
import torch


def mnist_data_loader(train, batch_size):
    # 1st Dataloader for Source Dataset at Classification Domain TF
    dataset = torchvision.datasets.MNIST(root='/MNIST', train=train, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.Resize((32, 32)),
                                             torchvision.transforms.CenterCrop((32, 32)),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(mean=0.485, std=0.229)
                                         ]))
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [40000, 10000, 10000])
    data_load = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_load_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return data_load, data_load_valid


def svhn_data_loader(batch_size):
    # 2nd Dataloader for Target Dataset at Classification Domain TF
    # torchvision.transforms.Compose is from https://discuss.pytorch.org/t/change-3-channel-to-1-channel/46619/2
    train_dataset = torchvision.datasets.SVHN(root='/SVHN', split='train', download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.Grayscale(num_output_channels=1),
                                                  # torchvision.transforms.Resize((28, 28)),
                                                  torchvision.transforms.CenterCrop((32, 32)),
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(mean=0.485, std=0.229)]))
    test_dataset = torchvision.datasets.SVHN(root='/SVHN', split='test', download=True,
                                             transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.Grayscale(num_output_channels=1),
                                                  # torchvision.transforms.Resize((28, 28)),
                                                  torchvision.transforms.CenterCrop((32, 32)),
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(mean=0.485, std=0.229)]))
    train_load = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_load = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_load, test_load
