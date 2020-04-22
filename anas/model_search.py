import torch
import torch.nn as nn
from operations import *
from genotypes import ATT_PRIMITIVES



class att_MixedOp(nn.Module):
    def __init__(self, C, id_reg=False):
        super(att_MixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        self.p = 0.
        for i in range(len(ATT_PRIMITIVES)):
            primitive = ATT_PRIMITIVES[i]
            op = att_OPS[primitive](C, False)
            if isinstance(op, Identity) and id_reg:
                op = nn.Sequential(op, nn.Dropout(self.p))
            self.m_ops.append(op)

    def update_p(self):
        for op in self.m_ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], Identity):
                    op[1].p = self.p

    def forward(self, x, weights):
        return sum(w * a_op(x) for w, a_op in zip(weights, self.m_ops))


class a_Cell(nn.Module):
    def __init__(self, C, steps=3):
        super(a_Cell, self).__init__()
        self._steps = steps
        self.p = 0.
        self.cell_ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                op = att_MixedOp(C)
                self.cell_ops.append(op)
        self.activ = nn.Sigmoid()

    def update_p(self):
        for op in self.cell_ops:
            op.p = self.p
            op.update_p()

    def forward(self, x, weights):
        s0 = s1 = x
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self.cell_ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        mask = self.activ(sum(states[2:]))
        return x * mask
