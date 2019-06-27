import torch
import numpy as np
from PIL import Image
from glob import glob
from image.dataquantizer import quantize


def load_image(location):
    img = Image.open(location)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(image, location):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    img = Image.fromarray(np.asarray(np.clip(image, 0, 255), dtype="uint8"))
    img.save(location)


def load_imagenet():
    directory = "/home/xinyan/program/data/youtube_frames_720p/*bmp"
    # directory = "data/tiny-imagenet-200/train/*/images/*.JPEG"
    # directory = "/research/jcheng2/xinyan/zzhang/AlexnetandVGG/ILSVRC2012/train/*/*.JPEG"
    locations = glob(directory)[:100]
    print(locations)
    print(locations[:10])
    N = len(locations)
    first = load_image(locations[0])
    print(first.shape)
    vectors = np.empty(
        ([N] + list(first.shape)), dtype=np.float32
    )
    print(vectors.shape)
    for i, location in enumerate(locations):
        image = load_image(location)
        if image.shape == vectors[i, :].shape:
            vectors[i, :] = image.astype(np.float32)
        else:
            assert False
    return vectors


if __name__ == "__main__":
    vectors = load_imagenet()
    for i in range(10):
        save_image(vectors[i, :], 'original_{}.png'.format(i))
    compressed_rq = quantize(vectors,  m=1, depth=128)

    for i in range(10):
        save_image(compressed_rq[i, :], 'compressed_rq_{}.png'.format(i))

    compressed_pq = quantize(vectors,  m=128, depth=1)
    for i in range(10):
        save_image(compressed_pq[i, :], 'compressed_pq_{}.png'.format(i))
