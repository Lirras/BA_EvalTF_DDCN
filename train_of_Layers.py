import torch


def prepared_optimizer():
    optimizer = torch.optim.Adam(lr=0.01)
    crit = torch.nn.CrossEntropyLoss()


def conv2d(dataset, in_channels, out_channels, kernel_size, stride, padding, padding_mode, activation=None):
    x = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
    optimizer = torch.optim.Adam(x.parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()

    new_dataset = dataset
    counter = 0
    for data, label in dataset:
        optimizer.zero_grad()
        if activation == 'relu':
            output = torch.nn.functional.relu(x(data))
        else:
            output = x(data)
        output = torch.flatten(output, 1)
        loss = crit(output, label)
        loss.backward()
        optimizer.step()
        # new_dataset[counter] = output, label
        counter += 1
        if counter == len(dataset):
            break
    return new_dataset


def batch_norm(dataset):
    torch.nn.functional.batch_norm()
    return


def dense(dataset):
    torch.nn.Linear()
    return


def flatten(dataset):
    torch.flatten()
    return


def unflatten(dataset):
    torch.unflatten()
    return


def reshape(dataset):
    torch.reshape()
    return


def max_pooling(dataset):
    torch.nn.functional.max_pool2d()
    return


def squeeze(dataset):
    torch.squeeze()
    return


def unsqueeze(dataset):
    torch.unsqueeze()
    return


def softmax(dataset):
    torch.nn.functional.softmax()
    return
