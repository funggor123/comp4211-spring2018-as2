import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from encoder import Encoder
from predictor import Predictor
from decoder import Decoder
import random

use_gpu = torch.cuda.is_available()
use_gpu = True


def cut_validation(data_to_cut, ratio_of_train=0.8):
    train_size = int(ratio_of_train * len(data_to_cut))
    test_size = len(data_to_cut) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data_to_cut, [train_size, test_size])
    return train_dataset, test_dataset


def make_data_loader(data_to_loader, batch_size=16):
    data_loader = torch.utils.data.DataLoader(data_to_loader,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16)
    return data_loader


def show_image(image):
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()


def tune_encoder_params(train_set, vad_set, pre_trained_path=""):
    opts = [("ADAM", 0.001), ("SGD", 0.1), ("SGD", 0.01)]
    hid = [32, 64]
    epoch = 3
    best_loss = 99999999999999999
    best_accuracy = 0
    best_set_of_parameters = []
    rounds = 0

    print("Training Set Size =", len(train_set))
    print("Training Epochs =", epoch)
    print("Validation Set Size =", len(vad_set))

    for hidden_num in hid:
        for opt in opts:
            rounds += 1
            print("(Round " + str(rounds) + ") Parameters Details (opt,lr,hid):", opt[0],
                  "," + str(opt[1]) + " ," + str(hidden_num))
            net, predictor, test_loss, test_accuracy = train_encoder(train_set, epoch=epoch, learning_r=opt[1], hidden_num=hidden_num,
                                                                     opt=opt[0],
                                                                     pre_trained_path=pre_trained_path, test_set=vad_set)
            if test_loss < best_loss:
                best_loss = test_loss
                best_accuracy = test_accuracy
                best_set_of_parameters = [hidden_num, opt[0], opt[1]]
    print("Best Hyper-parameters obtains after the hold out validation [hid,opt,lr] ")
    print(best_set_of_parameters, "with ", best_accuracy, " of validation accuracy with Cross Entropy Loss: ", best_loss)
    return best_set_of_parameters


def test_encoder(model, predictor, vad_set, name="Validation", show_log=True):
    data_loader = make_data_loader(vad_set)
    correct = 0
    total = 0
    total_loss = 0

    criterion = nn.CrossEntropyLoss()

    for images, labels in data_loader:
        labels_cpu = labels
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        outputs = predictor(outputs)

        loss = criterion(outputs, labels)

        if use_gpu:
            _, predicted = torch.max(outputs.data.cpu(), 1)
        else:
            _, predicted = torch.max(outputs.data, 1)

        total += labels_cpu.size(0)
        correct += (predicted == labels_cpu).sum().item()
        total_loss += loss.item()

    if show_log:
        print(" " + name + ' Accuracy of the model on the ' + name + ' set images: {} %'.format(100 * correct / total),
          " with Cross Entropy Loss: ", total_loss)
    return total_loss, 100 * correct / total


def test_decoder(encoder, decoder, vad_set, name="Validation"):
    data_loader = make_data_loader(vad_set)
    total_loss = 0
    rounds = 0
    num_show_picture = 1
    num_show = 0

    criterion = nn.MSELoss()

    for images, labels in data_loader:
        if use_gpu:
            images = images.cuda()

        rounds = rounds + 1
        outputs = encoder(images)
        outputs = decoder(outputs)

        if num_show < num_show_picture and random.random() > 0.5:
            num_show += 1
            show_image(outputs.cpu().detach().numpy()[0][0])

        loss = criterion(outputs, images)
        total_loss += loss.item()

    print(name + ' MSE Loss of the model on the ' + name + ' set images: {}'.format(total_loss / rounds))
    return total_loss


def train_encoder(train_set, hidden_num, opt, learning_r, epoch=500, batch_size=32, pre_trained_path='', test_set=None, name="Validation"):
    data_loader = make_data_loader(data_to_loader=train_set, batch_size=batch_size)
    best_test_loss = 9999999999999
    best_test_accuracy = 0

    net = Encoder()
    predictor = Predictor(hidden_num=hidden_num)

    if pre_trained_path != '':
        net = torch.load(pre_trained_path)["model"]

    if use_gpu:
        net = net.cuda()
        predictor = predictor.cuda()

    criterion = nn.CrossEntropyLoss()
    params = list(net.parameters()) + list(predictor.parameters())
    if opt == 'ADAM':
        optimizer = optim.Adam(params, lr=learning_r)
    else:
        optimizer = optim.SGD(params, lr=learning_r)

    for epoch in range(epoch):  # loop over the dataset multiple times
        print("Epoch: ", epoch)
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs
            inputs, labels = data

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs = predictor(outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        if test_set is not None:
            test_loss, accuracy = test_encoder(net, predictor, test_set, name=name, show_log=True)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_accuracy = accuracy

    return net, predictor, best_test_loss, best_test_accuracy


def train_decoder(train_set, opt, learning_r, encoder=None, epoch=500, batch_size=32, pre_trained_path=''):
    data_loader = make_data_loader(data_to_loader=train_set, batch_size=batch_size)

    if encoder is None:
        encoder = Encoder()
    decoder = Decoder()

    if pre_trained_path != '':
        encoder = torch.load(pre_trained_path)["model"]

    if use_gpu:
        decoder = decoder.cuda()
        encoder = encoder.cuda()

    criterion = nn.MSELoss()
    params = list(decoder.parameters()) + list(encoder.parameters())

    if opt == "ADAM":

        optimizer = optim.Adam(params, lr=learning_r)
    else:
        optimizer = optim.SGD(params, lr=learning_r)

    for epoch in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs
            inputs, labels = data

            if use_gpu:
                inputs = inputs.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            hidden = encoder(inputs)
            outputs = decoder(hidden)

            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    return encoder, decoder


def tune_decoder_params(train_set, vad_set, pre_trained_path=""):
    opts = [("ADAM", 0.001), ("SGD", 0.1), ("SGD", 0.01)]
    epoch = 3
    best_accuracy = -1
    best_set_of_parameters = []
    rounds = 0

    print("---Parameters Tuning---")
    print("Training Set Size = ", len(train_set))
    print("Validation Set Size = ", len(vad_set))
    print("Epoch =", epoch)

    for opt in opts:
        rounds += 1
        print("(Round " + str(rounds) + ") Parameters Details (opt,lr):", opt[0],
              ", " + str(opt[1]))
        encoder, decoder = train_decoder(train_set, epoch=epoch, learning_r=opt[1],
                                         opt=opt[0],
                                         pre_trained_path=pre_trained_path)
        accuracy = test_decoder(encoder, decoder, vad_set)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_set_of_parameters = [opt[0], opt[1]]
    print("Best Hyper-parameters obtains after the hold out validation [hid,opt,lr] ")
    print(best_set_of_parameters)
    print("with ", best_accuracy, " of validation accuracy")
    return best_set_of_parameters
