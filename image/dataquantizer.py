import torch
import numpy as np
from torchvision import datasets, transforms
from quantizer.rq import ResidualPQ
from quantizer.pq import PQ


class QuantizedData(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], int(self.targets[index])

    def __len__(self):
        return len(self.data)


def save_image(image, location):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    from PIL import Image
    img = Image.fromarray(np.asarray(np.clip(image, 0, 255), dtype=np.uint8))
    img.save(location)


def normalize(data, mean, std):
    transformer = transforms.Normalize(mean, std, inplace=True)
    np_view = data.numpy() if isinstance(data, torch.Tensor) else data
    if np_view.ndim == 3:
        np_view = np_view[:, :, :, None]
    # convert from (H, W, C) to (C, H, W)
    tensor = torch.from_numpy(np_view.transpose((0, 3, 1, 2))).float().div(255)
    for t in tensor:
        transformer(t)
    return tensor


def quantize(data, m=1, depth=1, dim=None, transpose=False):
    np_view = data.numpy() if isinstance(data, torch.Tensor) else data

    if transpose:
        np_view = np_view.transpose((0, 2, 3, 1))  # convert from (C, H, W) to (H, W, C)
    shapes = np_view.shape

    if dim is None:
        dim = np_view.size // shapes[0] // shapes[1]
    assert np_view.size // shapes[0] % dim == 0

    xs = np_view.reshape(-1, dim).astype(np.float32)
    rq = ResidualPQ([PQ(M=m, Ks=256) for _ in range(depth)])
    compressed = rq.fit(xs, iter=20).compress(xs)
    images = compressed.reshape(shapes)
    if transpose:
        images = images.transpose((0, 3, 1, 2))  # convert from (H, W, C) to (C, H, W)
    return torch.from_numpy(images)


def minst(args):
    train_data = datasets.MNIST(
        './data', train=True, download=True, transform=None)
    normalized = normalize(train_data.data, (0.1307,), (0.3081,))
    quantized = quantize(normalized, m=args.m, depth=args.depth, transpose=True)
    quantized_data = QuantizedData(quantized, train_data.targets)
    train_loader = torch.utils.data.DataLoader(
        quantized_data, batch_size=args.batch_size, shuffle=True,)

    test_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_data = datasets.MNIST(
        './data', train=False, transform=test_transformer)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_batch_size, shuffle=True,)
    return train_loader, test_loader


def cifar10(args):
    train_data = datasets.CIFAR10(
        root='./data', train=True, download=True)
    normalized = normalize(train_data.data,
                           (0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    quantized = quantize(normalized, m=args.m, depth=args.depth, transpose=True)
    quantized_data = QuantizedData(quantized, train_data.targets)
    train_loader = torch.utils.data.DataLoader(
        quantized_data, batch_size=args.batch_size, shuffle=True, num_workers=2)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, args.test_batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def visual():
    vectors = datasets.CIFAR10('./data', train=True, download=True).data
    for i in range(10):
        save_image(vectors[i], 'cifar/original_{}.png'.format(i))
    compressed_rq = quantize(vectors, m=1, depth=16)
    for i in range(10):
        save_image(compressed_rq[i], 'cifar/compressed_rq_{}.png'.format(i))
    compressed_pq = quantize(vectors, m=16, depth=1)
    for i in range(10):
        save_image(compressed_pq[i], 'cifar/compressed_pq_{}.png'.format(i))


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    args = DotDict()
    args.batch_size = 256
    args.test_batch_size = 1000
    args.m = 1
    args.depth = 1
    minst(args=args)
