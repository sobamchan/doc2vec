import numpy as np
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L

import sobamchan_chainer

class NLMBase(sobamchan_chainer.Model):

    def __init__(self, vocab_num, D, doc_num):
        super(NLMBase, self).__init__(
            embed_word = L.EmbedID(vocab_num, D),
            embed_doc = L.EmbedID(doc_num, D),
            l = L.Linear(None, vocab_num),
        )

    def fwd(self, wk, dk):
        x = self.embed_word(wk)  # (1)
        d = self.embed_doc(dk)
        xd = F.concat([x, d], axis=0)
        x = self.l(xd.data.reshape(1, xd.shape[0], xd.shape[1]))  # (1)
        return x

    def __call__(self, dk, wk, wt):
        '''
        input:
            dk: document id
            wk: list of word ids
            wt: target word id
        output: prob
        '''
        yw = self.fwd(wk, dk)
        loss = F.softmax_cross_entropy(yw, wt)
        return loss
