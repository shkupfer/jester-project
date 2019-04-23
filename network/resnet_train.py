import argparse
import logging
import sys
import os
import math
import time
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from network.utils import JesterDatasetFolder, WidthPadder, accuracy_from_loader
from resnet import ResNet, BasicBlock

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_loader_workers = 8


def train_resnet(data_dir, delimiter, limit_to_n_imgs, width_extend, batch_size, num_epochs, learning_rate, mdl_file, plot_file, confusion_matrix_file):
    train_imgseqs_dir = os.path.join(data_dir, 'train_imgseqs/')
    test_imgseqs_dir = os.path.join(data_dir, 'test_imgseqs/')
    train_targets = os.path.join(data_dir, 'jester-v1-train.csv')
    test_targets = os.path.join(data_dir, 'jester-v1-validation.csv')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    WidthPadder(pad_to_width=width_extend)])  # This pads each image with gray pixels on the left and right, so that each is 176px wide

    train_dataset = JesterDatasetFolder(train_imgseqs_dir, train_targets, transform=transform, delimiter=delimiter, limit_to_n_imgs=limit_to_n_imgs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_loader_workers)

    test_dataset = JesterDatasetFolder(test_imgseqs_dir, test_targets, transform=transform, delimiter=delimiter, limit_to_n_imgs=limit_to_n_imgs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_loader_workers)

    rnet = nn.DataParallel(ResNet(BasicBlock, num_classes=27).to(device)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnet.parameters(), lr=learning_rate)

    train_acc_by_epoch, test_acc_by_epoch = [], []

    start_time = time.time()
    test_acc = -1
    for epoch in range(num_epochs):
        for i, (imgseqs, labels) in enumerate(train_loader):
            logger.info('Epoch %d/%d, Batch %d/%d' % (epoch + 1, num_epochs, i + 1, int(math.ceil(float(len(train_dataset)) / batch_size))))

            imgseqs = imgseqs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = rnet(imgseqs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            logger.info('Batch loss: %.4f' % loss.item())


        rnet.eval()

        logger.info("Saving model to %s" % mdl_file)
        torch.save(rnet.state_dict(), mdl_file)

        train_acc = accuracy_from_loader(rnet, train_loader, scale=100, return_data=False)
        logger.info("Accuracy on training set after %d epochs: %.2f%%" % (epoch + 1, train_acc))
        train_acc_by_epoch.append(train_acc)

        last_test_acc = test_acc

        test_acc, test_predicted, test_labels = accuracy_from_loader(rnet, test_loader, scale=100, return_data=True)
        logger.info("Accuracy on test set after %d epochs: %.2f%%" % (epoch + 1, test_acc))
        test_acc_by_epoch.append(test_acc)

        if test_acc < last_test_acc:
            logger.info("Test accuracy for epoch %d (%.2f%%) was less than previous epoch (%.2f%%), so stopping here and keeping the model from previous epoch" %
                        (epoch + 1, test_acc, last_test_acc))
            break

        rnet.train()
    end_time = time.time()
    logger.info("Training time: %s" % (end_time - start_time))

    plt.figure(1)
    plt.plot(range(1, num_epochs + 1), train_acc_by_epoch, label='Training Set')
    plt.plot(range(1, num_epochs + 1), test_acc_by_epoch, label='Testing Set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(0, num_epochs + 1, 2))
    plt.xlim(right=num_epochs + 0.5)
    plt.ylim(bottom=0, top=100)
    plt.legend()
    plt.savefig(plot_file)
    try:
        plt.show()
    except:
        logger.info("Unable to show plot")

    confusion_matrix = np.zeros((len(train_dataset.classes), len(train_dataset.classes)))

    for l, p in zip(test_labels, test_predicted):
        confusion_matrix[l.item(), p.item()] += 1

    conf_df = pd.DataFrame(confusion_matrix).astype(int)
    conf_df.index = conf_df.columns = [train_dataset.classes[i] for i in conf_df.index]
    logger.info("Confusion matrix (also saved to %s)\n(Each row is an actual class, each column is a predicted class):\n%s" % (confusion_matrix_file, conf_df))
    conf_df.to_csv(confusion_matrix_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
    help='Directory containing train and test image sequences (in train_imgseqs/ and test_imgseqs/) and targets (in jester-v1-train.csv and jester-v1-validation.csv)')
    parser.add_argument('--batch_size', '-b', type=int, default=150, required=True, help='Batch size')
    parser.add_argument('--delimiter', '-d', default=';', help='Delimiter for targets csv file')
    parser.add_argument('--limit_to_n_imgs', '-l', type=int, default=36, help='Limit (or increase) the number of images in each image sequence to this many')
    parser.add_argument('--width_extend', '-w', type=int, default=176, help='For images less than this width, pad with gray pixels on either side')
    parser.add_argument('--num_epochs', '-e', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='Learning rate. For now this uses an Adam optimizer')
    parser.add_argument('--mdl_file', '-f', default='model.pkl', help='Model will be saved in this file')
    parser.add_argument('--plot_file', '-p', default='acc_by_epoch.png', help='File to save accuracy graph')
    parser.add_argument('--confusion_matrix', '-cf', default='confusion_matrix.csv', help='Save a confusion matrix based on test outputs to this file')

    args = parser.parse_args()

    train_resnet(args.data_dir, args.delimiter, args.limit_to_n_imgs, args.width_extend, args.batch_size,
                 args.num_epochs, args.learning_rate, args.mdl_file, args.plot_file, args.confusion_matrix)
