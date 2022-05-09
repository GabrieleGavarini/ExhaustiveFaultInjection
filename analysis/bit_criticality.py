import pandas

from models.resnet import resnet20
from models.utils import load_from_dict
import struct

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from tqdm import tqdm


def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')


def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]


def binary_to_count(binary_number):
    return [int(bit) for bit in binary_number]

network = resnet20()
network_path = '../models/pretrained_models/resnet20-trained.th'
load_from_dict(network=network,
               device='cuda',
               path=network_path)


weights = [param.flatten().detach().numpy() for name, param in network.named_parameters() if (('conv' in name or 'linear' in name) and 'weight' in name)]
weights_binary = [np.flip([binary_to_count(float_to_bin(layer_weight)) for layer_weight in layer_weights]) for layer_weights in weights]
weights_binary = np.concatenate(weights_binary)
weights_binary_df = pd.DataFrame(weights_binary)

distance_list = []

for bit_index in np.arange(0, 32):
    bit_at_0_average_distance = 0
    bit_at_0 = weights_binary_df[weights_binary_df.iloc[:, bit_index] == 0]

    pbar = tqdm(bit_at_0.iterrows(), total=len(bit_at_0), desc=f'Bit {bit_index} at 0')
    for binary_index, binary_weight in enumerate(pbar):
        bit_list = binary_weight[1].astype(str).to_list()
        bit_str = ''.join(np.flip(bit_list))
        weight = bin_to_float(bit_str)

        binary_weight[1][bit_index] = 1
        faulty_bit_list = binary_weight[1].astype(str).to_list()
        faulty_bit_str = ''.join(np.flip(faulty_bit_list))
        faulty_weight = bin_to_float(faulty_bit_str)

        distance = np.abs(weight - faulty_weight)
        bit_at_0_average_distance = ((bit_at_0_average_distance * binary_index) + distance) / (binary_index + 1)
        pbar.set_postfix({'Distance': bit_at_0_average_distance})

    bit_at_1_average_distance = 0
    bit_at_1 = weights_binary_df[weights_binary_df.iloc[:, bit_index] == 1]

    pbar = tqdm(bit_at_1.iterrows(), total=len(bit_at_1), desc=f'Bit {bit_index} at 1')

    for binary_index, binary_weight in enumerate(pbar):
        bit_list = binary_weight[1].astype(str).to_list()
        bit_str = ''.join(np.flip(bit_list))
        weight = bin_to_float(bit_str)

        binary_weight[1][bit_index] = 0
        faulty_bit_list = binary_weight[1].astype(str).to_list()
        faulty_bit_str = ''.join(np.flip(faulty_bit_list))
        faulty_weight = bin_to_float(faulty_bit_str)

        distance = np.abs(weight - faulty_weight)
        bit_at_1_average_distance = ((bit_at_1_average_distance * binary_index) + distance) / (binary_index + 1)
        pbar.set_postfix({'Distance': bit_at_1_average_distance})

    distance_list.append({'Bit_0_Distance': bit_at_0_average_distance,
                          'Bit_1_Distance': bit_at_1_average_distance})

distance_df = pd.DataFrame(distance_list)
distance_df.to_csv('bit_distribution/criticality.csv')
