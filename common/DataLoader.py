import numpy as np


class DataLoader():
    def __init__(self, batch_size, seq_length, end_token=0):
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
        self.end_token = end_token

    def create_batches(self, data_file):
        self.token_stream = []

        with open(data_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.seq_length:
                    self.token_stream.append(parse_line[:self.seq_length])
                else:
                    while len(parse_line) < self.seq_length:
                        parse_line.append(self.end_token)
                    if len(parse_line) == self.seq_length:
                        self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)

        self.shuffle()

    def shuffle(self):
        shuffle_ids = np.arange(len(self.sequence_batch))
        np.random.shuffle(shuffle_ids)
        self.sequence_batch = np.array(self.sequence_batch)[shuffle_ids]
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
