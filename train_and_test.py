import torch


def train_epoch(dataset, model, device, crit, optimizer):
    model.to(device)
    counter = 1
    print(len(dataset))
    # print(len(model.list_of_layers))  # 3
    # print(get_last_layer_params(model.list_of_layers))  # 0
    start_value = get_last_layer_params(model.list_of_layers)
    for data, label in dataset:
        optimizer.zero_grad()
        output = data
        while start_value < len(model.list_of_layers):
            print(start_value)
            print(model.list_of_layers[start_value][0])
            output = model.list_of_layers[start_value][0](output)
            start_value += 1
        # Data through the whole Network
        # output = model(data.to(device))  # todo: There have to be a better Way
        loss = crit(output, label)
        loss.backward()
        optimizer.step()
        counter += 1
        # todo: this does exactly what i want, but it isn't enough!
        if counter == len(dataset):
            print(data.shape)
            print(output.shape)
            break
    print(counter)


def test_new_layer(test_dataset, model, device, loss_fn):
    # This function is from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    # The half is from me
    toggle, size = length_of_data(test_dataset.dataset, batch_size=100)  # from me
    num_batches = len(test_dataset) - toggle  # from me
    model.eval()
    counter = 1  # from me
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, label in test_dataset:
            output = model(data.to(device))  # from me
            test_loss += loss_fn(output, label).item()
            correct += (output.argmax(1) == label).type(torch.float).sum().item()
            counter += 1  # from me
            if counter == len(test_dataset):  # from me
                break  # from me
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def last_epoch(dataset, model, device, crit, optimizer):
    model.to(device)
    counter = 1
    start_value = len(model.list_of_layers) - get_last_layer_params(model.list_of_layers)
    new_dataset = []
    for data, label in dataset:
        optimizer.zero_grad()
        output = data
        # Data through the whole Network
        while start_value < len(model.list_of_layers):
            output = model.list_of_layers[start_value][0](output.to(device))  # todo: There must be real Layers!
            start_value += 1
        loss = crit(output, label)
        loss.backward()
        optimizer.step()
        counter += 1
        new_dataset.append((data, label))
        if counter == len(dataset):
            break
    return new_dataset


def length_of_data(length, batch_size):
    x = len(length) / batch_size
    y = 0
    if int(x) != x:
        print(int(x), x)
        y = 1
    return y, int(x) * batch_size


def get_last_layer_params(list_of_layers):
    # Backward Iteration for Seeking the Last Layer with Weight Parameters
    x = len(list_of_layers)-1
    while x >= 0:
        if list_of_layers[x][1] == 'relu' or list_of_layers[x][1] is None:
            return x
        else:
            x -= 1
    print("Error at get last Layer Params!")
    return -2  # Here is an Error
