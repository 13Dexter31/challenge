import numpy as np


class Dataset:
    def __init__(self, dataset):
        self.content = dataset

    def get_x(self):
        frag = self.content[:, -1]
        frag_count = len(frag)

        w, h, c = frag[0].shape

        x = np.empty([frag_count, w, h, c])
        for i in range(frag_count):
            x[i] = frag[i]

        return _normalize(x)

    def get_y(self):
        results = self.content[:, -2]
        result_length = len(results)

        y = np.empty((result_length, 1))
        for i in range(result_length):
            y[i] = results[i]


def _normalize(x):
    return x / 255
