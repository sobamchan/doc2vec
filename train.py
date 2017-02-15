import numpy as np
from chainer import optimizers

from iterator import Iterator
from model import NLMBase
import argparse

import sobamchan_vocabulary 
import sobamchan_utility
utility = sobamchan_utility.Utility()
vocaburaly = sobamchan_vocabulary.Vocabulary()

def get_args():
    parser = argparse.ArgumentParser('doc2vec')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--embedding-size', dest='D', type=int, default=400)
    parser.add_argument('--window-size', dest='window_size', type=int, default=3)
    return parser.parse_args()

def train(args):
    texts = utility.readlines_from_filepath('./test_texts.txt')
    labels = utility.readlines_from_filepath('./test_labels.txt')
    for text in texts:
        vocaburaly.new(text)
    
    vocab_num = len(vocaburaly)
    D = args.D
    optimizer = optimizers.SGD()
    doc_num = len(labels)
    batch_size = 10
    texts_int = []
    for text in texts:
        text_int = []
        for word in text.strip().split():
            word = word.lower()
            text_int.append(vocaburaly.w2i[word])
        texts_int.append(text_int)
    labels_int = list(range(doc_num))
    window_size = args.window_size
    epoch = args.epoch


    model = NLMBase(vocab_num, D, doc_num)
    optimizer.setup(model)

    for e in range(epoch):
        iterator = Iterator(batch_size, texts_int, labels_int, window_size)
        for i in iterator:
            label = model.prepare_input([i[0]], dtype=np.int32)
            center = model.prepare_input([i[1]], dtype=np.int32)
            context = model.prepare_input(i[2], dtype=np.int32)

            model.cleargrads()
            loss = model(label, context, center)
            loss.backward()
            optimizer.update()
            print(loss.data)


















if __name__ == '__main__':
    args = get_args()
    train(args)
