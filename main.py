import os
import csv
import itertools
import numpy as np
from tqdm import tqdm, trange

import torch

import argparse

from models.utils import load_CIFAR10_datasets, load_from_dict
from models.resnet import resnet20

from BitFlipFI import BitFlipFI

def main(layer_start=0, layer_end=-1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)
    print(f'Running on device {device}')

    _, _, test_loader = load_CIFAR10_datasets(train_batch_size=1)

    image_per_class = 10
    selected_test_list = []
    image_class_counter = [0] * 10
    for test_image in test_loader:
        if image_class_counter[test_image[1]] < image_per_class:
            selected_test_list.append(test_image)
            image_class_counter[test_image[1]] += 1

    resnet = resnet20()
    resnet.to(device)
    load_from_dict(network=resnet,
                   device=device,
                   path='models/pretrained_models/resnet20-trained.th')

    resnet_layers = [m for m in resnet.modules() if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear)]
    if layer_end == -1:
        resnet_layers = resnet_layers[layer_start:]
    else:
        resnet_layers = resnet_layers[layer_start:layer_end]
    resnet_layers_shape = [layer.weight.shape for layer in resnet_layers]

    exhaustive_fault_injection(net=resnet,
                               net_name='resnet20',
                               net_layer_shape=resnet_layers_shape,
                               loader=selected_test_list,
                               device=device)


def exhaustive_fault_injection(net,
                               net_name,
                               net_layer_shape,
                               loader,
                               device):

    cwd = os.getcwd()
    os.makedirs(f'{cwd}/fault_injection/{net_name}', exist_ok=True)

    exhaustive_fault_list = []
    for layer, layer_shape in enumerate(tqdm(net_layer_shape, desc='Generating exhaustive fault list')):

        if len(layer_shape) == 4:
            k = np.arange(layer_shape[0])
            dim1 = np.arange(layer_shape[1])
            dim2 = np.arange(layer_shape[2])
            dim3 = np.arange(layer_shape[3])
            bits = np.arange(0, 32)

            exhaustive_fault_list = exhaustive_fault_list + list(itertools.product(*[[layer], k, dim1, dim2, dim3, bits]))
        else:
            k = np.arange(layer_shape[0])
            dim1 = np.arange(layer_shape[1])
            bits = np.arange(0, 32)

            exhaustive_fault_list = exhaustive_fault_list + list(itertools.product(*[[layer], k, dim1, bits]))

    with torch.set_grad_enabled(False):
        net.eval()

        if layer_end == -1:
            filename = f'{cwd}/fault_injection/{net_name}/{layer_start}-{len(net_layer_shape) + layer_start}_exhaustive_results.csv'
        else:
            filename = f'{cwd}/fault_injection/{net_name}/{layer_start}-{layer_end}_exhaustive_results.csv'

        with open(filename, 'w', newline='') as f_inj:
            writer_inj = csv.writer(f_inj)
            writer_inj.writerow(['Injection',
                                 'Layer',
                                 'ImageIndex',  # Image index
                                 'Top_1',
                                 'Top_2',
                                 'Top_3',
                                 'Top_4',
                                 'Top_5',
                                 'Golden',
                                 'Bit',
                                 'NoChange'])
            f_inj.flush()

            pbar = tqdm(exhaustive_fault_list, desc='Exhaustive fault injection campaign')
            for injection_index, fault in enumerate(pbar):
                layer = fault[0]
                bit = fault[-1]

                layer_index = layer - layer_start

                pbar.set_description(f'Exhaustive fault injection campaign (layer {layer_index + 1} of {len(net_layer_shape)})')

                pfi_model = BitFlipFI(net,
                                      fault_location=fault,
                                      batch_size=1,
                                      input_shape=[3, 32, 32],
                                      layer_types=["all"],
                                      use_cuda=(device == 'cuda'))

                corrupt_net = pfi_model.declare_weight_bit_flip()

                for image_index, data in enumerate(loader):
                    x, y_true = data
                    x, y_true = x.to(device), y_true.to(device)

                    y_pred = corrupt_net(x)

                    top_5 = torch.topk(y_pred, 5)

                    output_list = [injection_index,
                                   layer_index,
                                   image_index,
                                   int(top_5.indices[0][0]),
                                   int(top_5.indices[0][1]),
                                   int(top_5.indices[0][2]),
                                   int(top_5.indices[0][3]),
                                   int(top_5.indices[0][4]),
                                   int(y_true),
                                   bit,
                                   False]

                    writer_inj.writerow(output_list)
                    f_inj.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exhaustive fault injection')
    parser.add_argument('--layer-start', type=int, default=0,
                        help='From which layer to start the exhaustive fault injection campaign')
    parser.add_argument('--layer-end', type=int, default=-1,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()

    layer_start = args.layer_start
    layer_end = args.layer_end

    main(layer_start, layer_end)
