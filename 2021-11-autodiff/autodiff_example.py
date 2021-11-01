import gzip
import itertools as it
import sys
import urllib.request
from pathlib import Path
from typing import Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np

from autodiff import Array, Op, Tensor


class Model:
    def __init__(self):
        random = np.random.RandomState(3721)
        self.parameters = []
        self.lr = 0.1
        self.W0 = self.parameter(random.randn(28 * 28, 256) / np.sqrt(28 * 28))
        self.b0 = self.parameter(np.zeros(256))
        self.W1 = self.parameter(random.randn(256, 10) / np.sqrt(256))
        self.b1 = self.parameter(np.zeros(10))

    def parameter(self, value: Array) -> Tensor:
        tensor = Tensor.wrap(value.astype(np.float32), requires_grad=True)
        self.parameters.append(tensor)
        return tensor

    def step(self, x: Array, y: Array) -> float:
        for parameter in self.parameters:
            parameter.grad[...] = 0
        h = Op.matmul(x, self.W0)
        h = Op.add(h, Op.broadcast_to(self.b0, h.value.shape))
        h = Op.relu(h)
        h = Op.matmul(h, self.W1)
        h = Op.add(h, Op.broadcast_to(self.b1, h.value.shape))
        loss = Op.softmax_cross_entropy(h, y)
        loss.backward()
        for parameter in self.parameters:
            parameter.value -= self.lr * parameter.grad
        return float(loss.value)


def mnist_batches(batch_size: int) -> Iterator[Tuple[Array, Array]]:
    # Download & read dataset
    data_dir = Path("data/mnist")
    data_dir.mkdir(parents=True, exist_ok=True)

    def _load(url: str, name: str, offset: int) -> Array:
        dest = data_dir / name
        if not dest.exists():
            print(f"Downloading {url} -> {dest}", file=sys.stderr)
            with urllib.request.urlopen(url) as f:
                dest.write_bytes(f.read())
        with gzip.open(dest) as f:
            return np.frombuffer(f.read()[offset:], dtype=np.uint8)

    images_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    images = (
        _load(images_url, "images.gz", offset=16).reshape(-1, 28, 28).astype(np.float32)
        / 255
    )
    labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    labels = _load(labels_url, "labels.gz", offset=8)

    # Yield batches
    random = np.random.RandomState(9741)
    while True:
        indices = np.arange(len(images))
        random.shuffle(indices)
        for offset in range(0, len(images) - batch_size + 1, batch_size):
            batch = indices[offset : offset + batch_size]
            yield images[batch], labels[batch]


model = Model()
losses = []
for batch_x, batch_y in it.islice(mnist_batches(batch_size=16), 2000):
    losses.append(model.step(batch_x.reshape(batch_x.shape[0], -1), batch_y))
plt.plot(np.convolve(losses, np.full(100, 1 / 100), mode="valid"))
plt.savefig("loss.png")
print(f"Final loss: {np.mean(losses[-100:])}")
