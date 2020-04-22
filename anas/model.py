import torch
import torch.nn as nn
from operations import *
from utils import drop_path


class a_Cell(nn.Module):
    def __init__(self, genotype, C):
        super(a_Cell, self).__init__()
        op_names, indices = zip(*genotype.att)
        self._compile(C, op_names, indices)
        self.activ = nn.Sigmoid()

    def _compile(self, C, op_names, indices):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = att_OPS[name](C, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, x, drop_prob):
        s0 = s1 = x

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
            mask = self.activ(sum(states[2:]))
        return x * mask
