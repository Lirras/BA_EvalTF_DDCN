import torch


def train_epoch(dataset, model, device, crit, optimizer):
    model.to(device)
    counter = 1
    print(len(dataset))
    for data, label in dataset:
        optimizer.zero_grad()
        # Data through the whole Network
        output = model(data.to(device))
        loss = crit(output, label.to(device))
        loss.backward()
        optimizer.step()
        counter += 1
        # todo: this does exactly what i want, but it isnt enough!
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
            test_loss += loss_fn(output, label.to(device)).item()
            correct += (output.argmax(1) == label.to(device)).type(torch.float).sum().item()
            counter += 1  # from me
            if counter == len(test_dataset):  # from me
                break  # from me
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def length_of_data(length, batch_size):
    x = len(length) / batch_size
    y = 0
    if int(x) != x:
        print(int(x), x)
        y = 1
    return y, int(x) * batch_size
