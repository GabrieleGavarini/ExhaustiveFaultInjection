from models.resnet import resnet20
from models.utils import load_from_dict
import struct

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))


def binary_to_count(binary_number):
    return [int(bit) for bit in binary_number]


network = resnet20()
network_path = '../models/pretrained_models/resnet20-trained.th'
load_from_dict(network=network,
               device='cuda',
               path=network_path)


weights = [param.flatten().detach().numpy() for name, param in network.named_parameters() if (('conv' in name or 'linear' in name) and 'weight' in name)]
weights_binary = [np.array([binary_to_count(binary(layer_weight)) for layer_weight in layer_weights]) for layer_weights in weights]
weights_binary_count = np.array([weight_binary.sum(axis=0) for weight_binary in weights_binary])

layer_weights_number = np.array([len(layer_weight) for layer_weight in weights])

weights_bit_to_1_percentage = np.flip(weights_binary_count.sum(axis=0) / layer_weights_number.sum(axis=0))
weights_bit_to_1_percentage_df = pd.DataFrame(weights_bit_to_1_percentage)
weights_bit_to_1_percentage_df.to_csv('bit_distribution/weights_bit_to_1_percentage_df.csv')

weights_bit_to_1_percentage_layer = np.array([np.flip(bit_count / layer_weights_number[layer_index]) for layer_index, bit_count in enumerate(weights_binary_count)])
weights_bit_to_1_percentage_layer_df = pd.DataFrame(weights_bit_to_1_percentage_layer)
weights_bit_to_1_percentage_layer_df.to_csv('bit_distribution/weights_bit_to_1_percentage_layer_df.csv')


# plt.bar(np.arange(0, 32), np.maximum(0, 0.5 - weights_bit_to_1_percentage))
# plt.xticks(np.arange(0, 32), labels=np.arange(0, 32), rotation=90)
# # plt.yscale('log')
# plt.show()
