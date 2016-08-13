import numpy as np
import argparse
import chainer
from chainer import cuda, Variable, serializers
from net import *
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--output', '-o', type=str)
args = parser.parse_args()

model = FastStyleNet()
serializers.load_npz(args.model, model)

def swap(A):
    B = np.einsum('ijkl->klji', A)
    return B

def get_conv(layer):
    layer_dict = {
        'W': swap(layer.W.data),
        'b': layer.b.data
    }
    return layer_dict

def get_bat(layer):
    layer_dict = {
        'gamma': layer.gamma.data,
        'beta': layer.beta.data
    }
    return layer_dict

def get_deconv(layer):
    layer_dict = {
        'W': swap(layer.W.data),
        'b': layer.b.data
    }
    return layer_dict

def get_res(layer):
    layer_dict = {
        'c1': get_conv(layer.c1),
        'c2': get_conv(layer.c2),
        'b1': get_bat(layer.b1),
        'b2': get_bat(layer.b2)
    }
    return layer_dict

model_dict = {
    'c1': get_conv(model.c1),
    'c2': get_conv(model.c2),
    'c3': get_conv(model.c3),
    'r1': get_res(model.r1),
    'r2': get_res(model.r2),
    'r3': get_res(model.r3),
    'r4': get_res(model.r4),
    'r5': get_res(model.r5),
    'b1': get_bat(model.b1),
    'b2': get_bat(model.b2),
    'b3': get_bat(model.b3),
    'b4': get_bat(model.b4),
    'b5': get_bat(model.b5),
    'd1': get_deconv(model.d1),
    'd2': get_deconv(model.d2),
    'd3': get_deconv(model.d3)
    }
    

f = open(args.output, 'wt')
pickle.dump(model_dict, f)
f.close()


