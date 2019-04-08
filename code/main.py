'''

Data Split
Use train_dataset and eval_dataset as train / test sets

'''
from torchvision.datasets import EMNIST
from torch.utils.data import ConcatDataset, Subset
from torchvision.transforms import ToTensor, Compose
import numpy as np
import train


# For convenience, show image at index in dataset
def show_image(dataset, index):
    import matplotlib.pyplot as plt
    plt.imshow(dataset[index][0][0], cmap=plt.get_cmap('gray'))


def get_datasets(split='balanced', save=False):
    download_folder = './data'

    transform = Compose([ToTensor()])

    dataset = ConcatDataset([EMNIST(root=download_folder, split=split, download=True, train=False, transform=transform),
                             EMNIST(root=download_folder, split=split, download=True, train=True, transform=transform)])

    # Ignore the code below with argument 'save'
    if save:
        random_seed = 4211  # do not change
        n_samples = len(dataset)
        eval_size = 0.2
        indices = list(range(n_samples))
        split = int(np.floor(eval_size * n_samples))

        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_indices, eval_indices = indices[split:], indices[:split]

        # cut to half
        train_indices = train_indices[:len(train_indices) // 2]
        eval_indices = eval_indices[:len(eval_indices) // 2]

        np.savez('train_test_split.npz', train=train_indices, test=eval_indices)

    # just use save=False for students
    # load train test split indices
    else:
        with np.load('./train_test_split.npz') as f:
            train_indices = f['train']
            eval_indices = f['test']

    train_dataset = Subset(dataset, indices=train_indices)
    eval_dataset = Subset(dataset, indices=eval_indices)

    return train_dataset, eval_dataset


# TODO 1. build your own CNN classifier with the given structure. DO NOT COPY OR USE ANY TRICK 2. load pretrained
#  encoder from 'pretrained_encoder.pt' and build a CNN classifier on top of the encoder 3. load pretrained encoder
#  from 'pretrained_encoder.pt' and build a Convolutional Autoencoder on top of the encoder (just need to implement
#  decoder) *** Note that all the above tasks include implementation, training, analyzing, and reporting

# example main code
# each img has size (1, 28, 28) and each label is in {0, ..., 46}, a total of 47 classes
if __name__ == '__main__':
    train_ds, eval_ds = get_datasets()

    img_index = 10
    show_image(train_ds, img_index)
    show_image(eval_ds, img_index)

    ''' Show some Details about the data'''
    print("Training set size : ", len(train_ds))
    print("Test Set size :", len(eval_ds))
    best_set_parameters = [32, "ADAM", 0.001]
    tune = True
    encoder_scratch = True
    encoder_pretrained = False
    decoder = True
    train_round = 1
    top_accuracy = []

    '''
    a. CNN Encoder from scratch
    '''
    if encoder_scratch:
        print("---A. Encoder Scratch---")
        if tune:
            '''
            1. Hold out Validation
            '''
            print("---1. Hold out Validation---")
            train_set, valid_set = train.cut_validation(train_ds)
            best_set_parameters = train.tune_encoder_params(train_set, valid_set)
        for i in range(train_round):
            '''
            2. Training from Scratch
            '''
            print("---2. Training Entire --")
            print("Round: ", i + 1)
            net, predictor, test_loss, test_accuracy = train.train_encoder(train_ds, hidden_num=best_set_parameters[0],
                                                                           opt=best_set_parameters[1],
                                                                           learning_r=best_set_parameters[2],
                                                                           epoch=10, test_set=eval_ds, name="Test")

    print("---------------------------------------------------------------------------------------------")
    '''
    b. CNN Encoder from Pre-Trained
    '''
    if encoder_pretrained:
        print("---B. Encoder Pretrained---")
        if tune:
            '''
            1. Hold out Validation
            '''
            print("---1. Hold out Validation---")
            train_set, valid_set = train.cut_validation(train_ds)
            best_set_parameters = train.tune_encoder_params(train_set, valid_set,
                                                            pre_trained_path='./pretrained_encoder.pt')
        for i in range(train_round):
            '''
            2. Training from Scratch
            '''
            print("---2. Training Entire --")
            print("Round: ", i + 1)
            net, predictor = train.train_encoder(train_ds, hidden_num=best_set_parameters[0],
                                                 opt=best_set_parameters[1],
                                                 learning_r=best_set_parameters[2],
                                                 epoch=10, pre_trained_path='./pretrained_encoder.pt')
            train.test_encoder(net, predictor, train_ds, "Training")
            '''
            3. Testing
            '''
            print("---3. Testing----")
            train.test_encoder(net, predictor, eval_ds, "Testing")
    print("---------------------------------------------------------------------------------------------")
    '''
    c. CNN Decoder
    '''
    if decoder:
        print("---C. Decoder ---")
        best_set_parameters = ["ADAM", 0.001]
        if tune:
            '''
            1. Hold out Validation
            '''
            print("---1. Hold out Validation---")
            train_set, valid_set = train.cut_validation(train_ds)
            best_set_parameters = train.tune_decoder_params(train_set, valid_set,
                                                            pre_trained_path='./pretrained_encoder.pt')
        '''
        2. Training from Scratch
        '''
        print("---2. Training Entire---")
        encoder, decoder = train.train_decoder(train_ds, opt=best_set_parameters[0],
                                               learning_r=best_set_parameters[1],
                                               epoch=2, pre_trained_path='./pretrained_encoder.pt')
        train.test_decoder(encoder, decoder, train_ds, "Training")
        '''
        3. Testing
        '''
        print("---3. Testing----")
        train.test_decoder(encoder, decoder, eval_ds, "Testing")
