import numpy as np
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L

import sobamchan_chainer

class NLMBase(sobamchan_chainer.Model):

    def __init__(self, vocab_num, D, output_num):
        super(NLMBase, self).__init__(
            embed = L.EmbedID(vocab_num, D),
            l = L.Linear(None, vocab_num),
        )

    def fwd(self, wk):
        x = self.embed(wk)  # (1)
        x = self.l(x)  # (1)
        return x

    def __call__(self, wk, wt):
        '''
        input:
            wk: list of word ids
            wt: target word id
        output: prob
        '''
        yw = self.fwd(wk)
        loss = F.softmax_cross_entropy(yw, wt)
        return loss
