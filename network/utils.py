from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS, has_file_allowed_extension, pil_loader, accimage_loader
import torch
from torch.nn.modules.module import Module
import os
import csv
import random
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_from_loader(model, dataloader, scale=1, return_data=False):
    all_predictions, all_labels = [], []
    for _, (imgseqs, labels) in enumerate(dataloader):
        imgseqs = imgseqs.to(device)
        labels = labels.to(device)

        outputs = model(imgseqs)

        _, predicted_classes = torch.max(outputs, 1)

        all_predictions.append(predicted_classes)
        all_labels.append(labels)

    all_predictions, all_labels = torch.cat(all_predictions), torch.cat(all_labels)
    correct = (all_predictions == all_labels).sum().item()
    acc = scale * float(correct) / all_predictions.shape[0]

    if return_data:
        return acc, all_predictions, all_labels
    return acc


def imgseq_loader(img_paths):
    from torchvision import get_image_backend
    try:
        if get_image_backend() == 'accimage':
            return [accimage_loader(path) for path in img_paths]
        else:
            return [pil_loader(path) for path in img_paths]
    except Exception as exp:
        print("Exception loading an image in %s:\n%s" % (os.path.dirname(img_paths[0]), str(exp)))
        raise exp


def orderedSampleWithoutReplacement(seq, k):
    if not 0<=k<=len(seq):
        raise ValueError('Required that 0 <= sample_size <= population_size')

    numbersPicked = 0
    for i,number in enumerate(seq):
        prob = (k-numbersPicked) / float((len(seq)-i))
        if random.random() < prob:
            yield number
            numbersPicked += 1


class WidthPadder(Module):
    def __init__(self, pad_dim=2, pad_to_width=176, value=0):
        super(WidthPadder, self).__init__()
        self.pad_dim = pad_dim
        self.pad_to_width = pad_to_width
        self.value = value

    def forward(self, input):
        need_to_pad = self.pad_to_width - input.shape[2]
        if need_to_pad & 0x1:
            raise Exception("Padding needs to be even on both sides")
        pad_each_side = need_to_pad / 2
        return torch.nn.functional.pad(input,
                                       (pad_each_side, pad_each_side, 0, 0, 0, 0),
                                       "constant",
                                       self.value)


class JesterDatasetFolder(DatasetFolder):
    def __init__(self, imgseqs_root, targets_csv_path, limit_to_n_imgs=37, delimiter=';', transform=None,
                 target_transform=None, loader=imgseq_loader, extensions=IMG_EXTENSIONS):
        with open(targets_csv_path, 'r') as targets_csv_file:
            targets_reader = csv.reader(targets_csv_file, delimiter=delimiter)
            targets_by_dir = {dirname: clas for (dirname, clas) in targets_reader}

        classes = sorted(set(targets_by_dir.values()))
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        samples = []
        imgseqs_root = os.path.expanduser(imgseqs_root)

        for imgseq_dir in sorted(os.listdir(imgseqs_root)):
            target = targets_by_dir.get(imgseq_dir, None)

            d = os.path.join(imgseqs_root, imgseq_dir)
            if not os.path.isdir(d):
                print("Directory %s does not exist" % d)
                continue

            if target is None:
                print("Image sequence %s is not in targets file %s" % (imgseq_dir, targets_csv_path))
                continue
            imgseq = []
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        imgseq.append(path)

            samples.append((imgseq, class_to_idx[target]))

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + imgseqs_root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        elif len(targets_by_dir) != len(samples):
            raise Exception("Number of targets (%s) does not match number of image sequences (%s)" % (len(targets_by_dir), len(samples)))

        self.root = imgseqs_root
        self.loader = loader
        self.extensions = extensions
        self.limit_to_n_imgs = limit_to_n_imgs

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        if self.transform is not None:
            sample = [self.transform(x) for x in self.loader(path)]
        else:
            sample = self.loader(path)

        # If there are more than limit_to_n_imgs images, keep only that many of them (while retaining their order)
        if len(sample) > self.limit_to_n_imgs:
            sample = orderedSampleWithoutReplacement(sample, self.limit_to_n_imgs)
        # If there are less than limit_to_n_imgs images, make it so that it has limit_to_n_imgs images
        # Half of the additional images are identical to the first and added to the beginning, and half are identical to the last and added to the end
        elif len(sample) < self.limit_to_n_imgs:
            diff = self.limit_to_n_imgs - len(sample)
            add_to_front = int(math.floor(diff / 2.))
            add_to_back = int(math.ceil(diff / 2.))
            sample = add_to_front * [sample[0]] + sample + add_to_back * [sample[-1]]
            # sample += (self.limit_to_n_imgs - len(sample)) * [torch.zeros_like(sample[0])]

        sample = torch.stack(list(sample)).permute(1, 0, 2, 3)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return super(JesterDatasetFolder, self).__len__()

    def __repr__(self):
        return super(JesterDatasetFolder, self).__repr__()
