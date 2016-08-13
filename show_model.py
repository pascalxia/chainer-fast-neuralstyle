import numpy as np
import argparse
import chainer
from chainer import cuda, Variable, serializers
from net import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str)
args = parser.parse_args()

model = FastStyleNet()
serializers.load_npz(args.model, model)

print model.r1.c1.W.data.shape
print model.b1.gamma.data
print model.b1.beta.data
print model.b3.decay
print model.b3.eps
