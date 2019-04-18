import os
import argparse

import torch
import torchvision.transforms as transforms

from network.utils import JesterDatasetFolder, WidthPadder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_loader_workers = 1


def test_data_load(path_to_jpgs, targets_csv_file, delimiter, limit_to_n_imgs, width_extend, batch_size, num_imgseqs):
    img_fnames = os.listdir(path_to_jpgs)
    print_this_many = 10
    print("First %s image dirnames: %s" % (print_this_many, img_fnames[:print_this_many]))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    WidthPadder(pad_to_width=width_extend)])  # This pads each image with white pixels on the left and right, so that each is 176px wide

    train_dataset = JesterDatasetFolder(path_to_jpgs, targets_csv_file, transform=transform, delimiter=delimiter, limit_to_n_imgs=limit_to_n_imgs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_loader_workers)

    for i, (imgseqs, labels) in enumerate(train_loader):
        if num_imgseqs is not None and i >= num_imgseqs:
            break
        imgseqs, labels = imgseqs.to(device), labels.to(device)
        print("Batch %s" % i)

        print("Image sequences batch shape: %s" % str(imgseqs.size()))  # [batch_size, num_imgs, color_channels, height, width]
        print("Labels: %s" % str(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_jpgs', help='Directory containing one directory per image sequence')
    parser.add_argument('targets_csv_file', help='CSV with no header, each line should have index;targetclass')
    parser.add_argument('--delimiter', '-d', default=';', help='Delimiter for targets csv file')
    parser.add_argument('--limit_to_n_imgs', '-l', type=int, default=4, help='Limit (or increase) the number of images in each image sequence to this many')
    parser.add_argument('--width_extend', '-w', type=int, default=5, help='For images less than this width, pad with white pixels on either side')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--num_imgseqs', '-n', type=int, default=None, help='How many batches of image sequences to load')

    args = parser.parse_args()

    test_data_load(args.path_to_jpgs, args.targets_csv_file, args.delimiter, args.limit_to_n_imgs, args.width_extend, args.batch_size, args.num_imgseqs)
