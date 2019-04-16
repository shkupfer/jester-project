import argparse
import logging
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from network.utils import JesterDatasetFolder, WidthPadder
from resnet import ResNet, BasicBlock

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_loader_workers = 4


def train_resnet(path_to_jpgs, targets_csv_file, delimiter, limit_to_n_imgs, width_extend, batch_size, num_epochs, learning_rate):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    WidthPadder(pad_to_width=width_extend)])  # This pads each image with gray pixels on the left and right, so that each is 176px wide

    train_dataset = JesterDatasetFolder(path_to_jpgs, targets_csv_file, transform=transform, delimiter=delimiter, limit_to_n_imgs=limit_to_n_imgs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_loader_workers)

    rnet = nn.DataParallel(ResNet(BasicBlock, num_classes=27).to(device)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnet.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(rnet.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (imgseqs, labels) in enumerate(train_loader):
            logging.info("Batch data shape: %s" % str(imgseqs.shape))
            imgseqs = imgseqs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = rnet(imgseqs)
            logging.info("Got outputs. Shape: %s. Calculating loss" % str(outputs.shape))
            loss = criterion(outputs, labels)
            logging.info("Calculated loss, now backpropagating and updating weights")
            loss.backward()
            optimizer.step()

            logging.info('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))

            _, predicted_classes = torch.max(outputs, 1)
            correct = (predicted_classes == labels).sum()
            total = labels.shape[0]
            acc = float(correct) / total
            logging.info("Accuracy on this batch: %s" % acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_jpgs', help='Directory containing one directory per image sequence')
    parser.add_argument('targets_csv_file', help='CSV with no header, each line should have index;targetclass')
    parser.add_argument('--batch_size', '-b', type=int, default=150, required=True, help='Batch size')
    parser.add_argument('--delimiter', '-d', default=';', help='Delimiter for targets csv file')
    parser.add_argument('--limit_to_n_imgs', '-l', type=int, default=36, help='Limit (or increase) the number of images in each image sequence to this many')
    parser.add_argument('--width_extend', '-w', type=int, default=176, help='For images less than this width, pad with gray pixels on either side')
    parser.add_argument('--num_epochs', '-e', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='Learning rate. For now this uses an Adam optimizer')

    args = parser.parse_args()

    train_resnet(args.path_to_jpgs, args.targets_csv_file, args.delimiter, args.limit_to_n_imgs, args.width_extend,
                 args.batch_size, args.num_epochs, args.learning_rate)
