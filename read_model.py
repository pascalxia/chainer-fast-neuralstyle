import numpy as np
import pickle

f = open('./model.pkl', 'rb')
model_dict = pickle.load(f)
f.close()
print model_dict
