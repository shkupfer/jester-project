import argparse
import logging
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from network.utils import JesterDatasetFolder, WidthPadder

import ipdb

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_loader_workers = 8

num_epochs = 10
learning_rate = 0.001
imgseq_size = [37, 100, 176]  # D, H, W


def train_cnn(path_to_jpgs, targets_csv_file, delimiter, limit_to_n_imgs, width_extend, batch_size):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    WidthPadder(pad_to_width=width_extend)])  # This pads each image with white pixels on the left and right, so that each is 176px wide

    train_dataset = JesterDatasetFolder(path_to_jpgs, targets_csv_file, transform=transform, delimiter=delimiter, limit_to_n_imgs=limit_to_n_imgs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_loader_workers)


    class CNN(nn.Module):
        def __init__(self, ks_1, ksize_1, pool_ksize_1, ks_2, ksize_2, pool_ksize_2, ks_3, ksize_3, pool_ksize_3, ipdbs=False):
            super(CNN, self).__init__()
            self.ipdbs = ipdbs
            self.layer1 = nn.Sequential(nn.DataParallel(nn.Conv3d(3, ks_1, kernel_size=ksize_1, padding=0)),
                                        nn.DataParallel(nn.BatchNorm3d(ks_1)),
                                        nn.DataParallel(nn.ReLU()),
                                        nn.DataParallel(nn.MaxPool3d(pool_ksize_1)))
            self.layer2 = nn.Sequential(nn.DataParallel(nn.Conv3d(ks_1, ks_2, kernel_size=ksize_2, padding=0)),
                                        nn.DataParallel(nn.BatchNorm3d(ks_2)),
                                        nn.DataParallel(nn.ReLU()),
                                        nn.DataParallel(nn.MaxPool3d(pool_ksize_2)))
            self.layer3 = nn.Sequential(nn.DataParallel(nn.Conv3d(ks_2, ks_3, kernel_size=ksize_3, padding=0)),
                                        nn.DataParallel(nn.BatchNorm3d(ks_3)),
                                        nn.DataParallel(nn.ReLU()),
                                        nn.DataParallel(nn.MaxPool3d(pool_ksize_3)))
            out_size = imgseq_size
            for ks, ksize, pool_ksize in [(ks_1, ksize_1, pool_ksize_1), (ks_2, ksize_2, pool_ksize_2), (ks_3, ksize_3, pool_ksize_3)]:
            # for ks, ksize, pool_ksize in [(ks_1, ksize_1, pool_ksize_1), (ks_2, ksize_2, pool_ksize_2)]:
                for dim in [0, 1, 2]:
                    out_size[dim] = out_size[dim] - ksize[dim] + 1
                    out_size[dim] = out_size[dim] / pool_ksize[dim]  # Floor division

            n_linear_outs = out_size[0] * out_size[1] * out_size[2] * ks_3
            # n_linear_outs = out_size[0] * out_size[1] * out_size[2] * ks_2

            self.fc = nn.DataParallel(nn.Linear(n_linear_outs, len(train_dataset.classes)))

        def forward(self, x):
            logger.info("Starting forward()")
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            logging.info("Done with forward()")
            return out

    cnn = CNN(ks_1=50, ksize_1=(3, 10, 16), pool_ksize_1=(2, 2, 2),
              ks_2=50, ksize_2=(3, 10, 16), pool_ksize_2=(2, 2, 2),
              ks_3=20, ksize_3=(2, 4, 6), pool_ksize_3=(1, 3, 5)).to(device)
    # cnn = CNN(ks_1=50, ksize_1=(4, 10, 16), pool_ksize_1=(2, 3, 3),
    #           ks_2=50, ksize_2=(4, 10, 16), pool_ksize_2=(2, 3, 3),
    #           ks_3=None, ksize_3=None, pool_ksize_3=None).to(device)

    for ind, layer in enumerate(list(cnn.children())):
        layer_children = list(layer.children())
        first_child = list(layer_children)[0]

        print("Layer %s (%s):\nWeight shape: %s, bias shape: %s" % (ind + 1,
                   first_child,
                   first_child.weight.shape if hasattr(first_child, 'weight') else 'N/A',
                   first_child.bias.shape if hasattr(first_child, 'bias') else 'N/A'))

    # ipdb.set_trace()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (imgseqs, labels) in enumerate(train_loader):
            logging.info("In epoch loop. Batch data shape: %s" % str(imgseqs.shape))
            imgseqs = imgseqs.to(device)
            labels = labels.to(device)
            logging.info("Converted images and labels to CUDA")

            optimizer.zero_grad()
            logging.info("About to run through CNN")
            outputs = cnn(imgseqs)
            logging.info("Got outputs. Shape: %s. Calculating loss" % str(outputs.shape))
            loss = criterion(outputs, labels)
            logging.info("Calculated loss, now backpropagating")
            loss.backward()
            logging.info("Now updating weights with optimizer")
            optimizer.step()

            logging.info('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))

            _, predicted_classes = torch.max(outputs, 1)
            correct = (predicted_classes == labels).sum()
            total = labels.shape[0]
            acc = float(correct) / total
            logging.info("Accuracy on this batch: %s" % acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument()
    parser.add_argument('path_to_jpgs', help='Directory containing one directory per image sequence')
    parser.add_argument('targets_csv_file', help='CSV with no header, each line should have index;targetclass')

    parser.add_argument('--batch_size', '-b', type=int, required=True, help='Batch size')

    parser.add_argument('--delimiter', '-d', default=';', help='Delimiter for targets csv file')
    parser.add_argument('--limit_to_n_imgs', '-l', type=int, default=37, help='Limit (or increase) the number of images in each image sequence to this many')
    parser.add_argument('--width_extend', '-w', type=int, default=176, help='For images less than this width, pad with gray pixels on either side')

    args = parser.parse_args()

    train_cnn(args.path_to_jpgs, args.targets_csv_file, args.delimiter, args.limit_to_n_imgs, args.width_extend, args.batch_size)
