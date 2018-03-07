import numpy as np

class BatchIterator(object):
    def __init__(self, data, batch_size, epoch=None, shuffle=False, mode='train'):
        if isinstance(data, tuple) or isinstance(data, list):
            self.data = [np.array(d) for d in data]
            self._size = self.data[0].shape[0]
        else:
            self.data = np.array(data)
            self._size = self.data.shape[0]
        self.indexs = np.arange(self._size)

        if shuffle:
            self.indexs = np.random.shuffle(self.indexs)

        self.stop = epoch
        self._index = 0
        self._epoch_count = 0
        self.batch_size = batch_size
        self.mode = mode
        if self.mode != 'train':
            self.stop = 1

        if not self.stop and mode == 'train':
            self._iters = 1000000000
        else:
            self._iters = self.stop * self._size // self.batch_size
    
    @property
    def size(self):
        return self._size
    
    @property
    def iters(self):
        return self._iters

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode != 'train' and self._epoch_count >= self.stop:
            raise StopIteration
        if self.stop and self._epoch_count >= self.stop:
            raise StopIteration
        if self._index == self._size:
            raise StopIteration
        if self._index + self.batch_size >= self._size:
            self._epoch_count += 1
            if self.mode == 'train':
                indexs = np.concatenate((self.indexs[self._index:], self.indexs[:(self.batch_size-(self._size - self._index))]))
            else:
                indexs = self.indexs[self._index:]
        else:
            indexs = self.indexs[self._index:self._index+self.batch_size]

        self._index = (self._index + self.batch_size) % self._size

        if isinstance(self.data, tuple) or isinstance(self.data, list):
            return [d[indexs, ...] for d in self.data]
        else:
            return self.data[indexs, ...]
