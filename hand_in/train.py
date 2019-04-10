import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from encoder import Encoder
from predictor import Predictor
from decoder import Decoder
import random
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

use_gpu = torch.cuda.is_available()
use_gpu = True
tune_params_epchs = 20

writer_train = SummaryWriter('runs/train_0')
writer_test = SummaryWriter('runs/test_0')


def cut_validation(data_to_cut, ratio_of_train=0.8):
    print("Split ratio: ", ratio_of_train)
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
    epoch = tune_params_epchs
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
            net, predictor, test_loss, test_accuracy, _ = train_encoder(train_set, epoch=epoch, learning_r=opt[1],
                                                                        hidden_num=hidden_num,
                                                                        opt=opt[0],
                                                                        pre_trained_path=pre_trained_path,
                                                                        test_set=vad_set)
            print("Optimal Testing Loss for Round- ", rounds, " is: ", test_loss, " with the optimal accuracy:", test_accuracy)
            if test_loss < best_loss:
                best_loss = test_loss
                best_accuracy = test_accuracy
                best_set_of_parameters = [hidden_num, opt[0], opt[1]]
    print("Best Hyper-parameters obtains after the hold out validation [hid,opt,lr] ")
    print(best_set_of_parameters, "with ", best_accuracy, " of validation accuracy with Cross Entropy Loss: ",
          best_loss, " in rounds ", rounds)
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


def test_decoder(encoder, decoder, vad_set, name="Validation", show_log=True, img_tag="", tensorboard=False, epoch=0):
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
            if tensorboard:
                raw = vutils.make_grid(images, normalize=True, scale_each=True)
                writer_test.add_image(img_tag + "Raw Image", raw, epoch)

                img = vutils.make_grid(outputs.cpu(), normalize=True, scale_each=True)
                writer_test.add_image(img_tag + "Reconstruct by CNN", img, epoch)

        loss = criterion(outputs, images)
        total_loss += loss.item()

    if show_log:
        print(" ", name + ' MSE Loss of the model on the ' + name + ' set images: {}'.format(total_loss / rounds))
    return total_loss / rounds


def train_encoder(train_set, hidden_num, opt, learning_r, epoch=500, batch_size=32, pre_trained_path='', test_set=None,
                  name="Validation", tensorboard=False):
    data_loader = make_data_loader(data_to_loader=train_set, batch_size=batch_size)
    best_test_loss = 9999999999999
    best_test_accuracy = 0
    best_train_accuracy = 0
    best_train_loss = 9999999999
    accuracy_array = []
    train_accuracy_array = []
    is_pretrained = False

    net = Encoder()
    predictor = Predictor(hidden_num=hidden_num)

    if pre_trained_path != '':
        net = torch.load(pre_trained_path)["model"]
        is_pretrained = True

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

        train_loss, accuracy = test_encoder(net, predictor, train_set, name="Training", show_log=True)
        train_accuracy_array.append(accuracy)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_accuracy = accuracy

        if test_set is not None:
            test_loss, accuracy = test_encoder(net, predictor, test_set, name=name, show_log=True)
            accuracy_array.append(accuracy)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_accuracy = accuracy

        if is_pretrained:
            name = "is_pretrained"
        else:
            name = "scratch"

        if tensorboard:
            writer_train.add_scalar('loss_'+name, best_train_loss, epoch)
            writer_test.add_scalar('losss_'+name, best_test_loss, epoch)

            writer_train.add_scalar('top-1_accuracys_'+name, best_train_accuracy, epoch)
            writer_test.add_scalar('top-1_accuracys_'+name, best_test_accuracy, epoch)

            if len(accuracy_array) < 3:
                writer_train.add_scalar('top-3_accuracys_'+name, 0, epoch)
                writer_test.add_scalar('top-3_accuracys_'+name, 0, epoch)
            else:
                writer_train.add_scalar('top-3_accuracys_'+name, sorted(train_accuracy_array)[-3], epoch)
                writer_test.add_scalar('top-3_accuracys_'+name, sorted(accuracy_array)[-3], epoch)

    return net, predictor, best_test_loss, best_test_accuracy, accuracy_array


def train_decoder(train_set, opt, learning_r, encoder=None, epoch=500, batch_size=32, pre_trained_path='',
                  test_set=None,
                  name="Validation", tensorboard=False, img_tag=""):
    data_loader = make_data_loader(data_to_loader=train_set, batch_size=batch_size)
    best_test_loss = 999999999

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
        print("Epoch: ", epoch)

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

        train_loss = test_decoder(encoder, decoder, vad_set=train_set, name="Training", show_log=True, img_tag=img_tag,
                                  tensorboard=tensorboard, epoch=epoch)

        if test_set is not None:
            test_loss = test_decoder(encoder, decoder, vad_set=test_set, name=name, show_log=True, img_tag=img_tag,
                                     tensorboard=tensorboard, epoch=epoch)
            if tensorboard and name is not "Validation":
                writer_test.add_scalar('MSE_loss', test_loss, epoch)

            if test_loss < best_test_loss:
                best_test_loss = test_loss

        if tensorboard and name is not "Validation":
            writer_train.add_scalar('MSE_loss', train_loss, epoch)

    return encoder, decoder, best_test_loss


def tune_decoder_params(train_set, vad_set, pre_trained_path=""):
    opts = [("ADAM", 0.001), ("SGD", 0.1), ("SGD", 0.01)]
    epoch = tune_params_epchs
    best_test_loss = 99999
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
        encoder, decoder, test_loss = train_decoder(train_set, epoch=epoch, learning_r=opt[1],
                                                    opt=opt[0],
                                                    pre_trained_path=pre_trained_path,
                                                    test_set=vad_set,
                                                    img_tag="Img_Results_Val_Round_"+str(rounds), tensorboard=True)
        print("Optimal Testing Loss for Round- ", rounds, " is: ", test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_set_of_parameters = [opt[0], opt[1]]
    print("Best Hyper-parameters obtains after the hold out validation [hid,opt,lr] ")
    print(best_set_of_parameters, " with ", best_test_loss, " of lowest validation loss in rounds", rounds)
    return best_set_of_parameters
