import os
import csv
import itertools
import numpy as np
from tqdm import tqdm

import torch

import argparse

from models.utils import load_CIFAR10_datasets, load_from_dict
from models.resnet import resnet20
from models.densenet import densenet121
from models.mobilenetv2 import MobileNetV2

from BitFlipFI import BitFlipFI


def main(layer_start=0, layer_end=-1, network_name='resnet20'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)
    print(f'Running on device {device}')

    _, _, test_loader = load_CIFAR10_datasets(test_batch_size=10, test_image_per_class=1)

    if network_name == 'resnet20':
        network = resnet20()
        network_path = 'models/pretrained_models/resnet20-trained.th'
    elif network_name == 'densenet121':
        network = densenet121()
        network_path = 'models/pretrained_models/densenet121.pt'
    elif network_name == 'mobilenet-v2':
        network = MobileNetV2()
        network_path = 'models/pretrained_models/mobilenetv2.pth'

    network.to(device)
    load_from_dict(network=network,
                   device=device,
                   path=network_path)

    network_layers = [m for m in network.modules() if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear)]
    if layer_end == -1:
        resnet_layers = network_layers[layer_start:]
    else:
        resnet_layers = network_layers[layer_start:layer_end]
    resnet_layers_shape = [layer.weight.shape for layer in resnet_layers]

    exhaustive_fault_injection(net=network,
                               net_name=network_name,
                               net_layer_shape=resnet_layers_shape,
                               loader=test_loader,
                               device=device,
                               layer_start=layer_start,
                               layer_end=layer_end)


def exhaustive_fault_injection(net,
                               net_name,
                               net_layer_shape,
                               loader,
                               device,
                               layer_start=0,
                               layer_end=-1):

    cwd = os.getcwd()
    os.makedirs(f'{cwd}/fault_injection/{net_name}', exist_ok=True)

    # exhaustive_fault_list = []
    # for layer, layer_shape in enumerate(tqdm(net_layer_shape, desc='Generating exhaustive fault list')):
    #
    #     if len(layer_shape) == 4:
    #         k = np.arange(layer_shape[0])
    #         dim1 = np.arange(layer_shape[1])
    #         dim2 = np.arange(layer_shape[2])
    #         dim3 = np.arange(layer_shape[3])
    #         bits = np.arange(0, 32)
    #
    #         exhaustive_fault_list = exhaustive_fault_list + list(itertools.product(*[[layer + layer_start], k, dim1, dim2, dim3, bits]))
    #     else:
    #         k = np.arange(layer_shape[0])
    #         dim1 = np.arange(layer_shape[1])
    #         bits = np.arange(0, 32)
    #
    #         exhaustive_fault_list = exhaustive_fault_list + list(itertools.product(*[[layer + layer_start], k, dim1, bits]))

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
                                 'ImageIndex',
                                 'Top_1',
                                 'Top_2',
                                 'Top_3',
                                 'Top_4',
                                 'Top_5',
                                 'Golden',
                                 'Bit',
                                 'NoChange'])
            f_inj.flush()

            injection_index = 0
            total = np.sum(np.array([np.prod(layer_shape) for layer_shape in net_layer_shape])) * 32
            pbar = tqdm(net_layer_shape, total=total)
            for layer, layer_shape in enumerate(pbar):
                for k in np.arange(layer_shape[0]):
                    for dim1 in np.arange(layer_shape[1]):
                        for dim2 in np.arange(layer_shape[2]):
                            for dim3 in np.arange(layer_shape[3]):
                                for bit in np.arange(0, 32):
                                    fault = [layer, k, dim1, dim2, dim3, bit]

                                    pbar.set_description(f'Exhaustive fault injection campaign (layer {layer} of {len(net_layer_shape) + layer_start - 1})')
                                    pbar.set_description(f'fault: {fault}')

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

                                        for index in range(0, len(top_5.indices)):
                                            output_list = [injection_index,
                                                           layer,
                                                           image_index * loader.batch_size + index,
                                                           int(top_5.indices[index][0]),
                                                           int(top_5.indices[index][1]),
                                                           int(top_5.indices[index][2]),
                                                           int(top_5.indices[index][3]),
                                                           int(top_5.indices[index][4]),
                                                           int(y_true[index]),
                                                           bit,
                                                           False]
                                            writer_inj.writerow(output_list)
                                            f_inj.flush()

                                    pbar.update(1)
                                    injection_index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exhaustive fault injection')
    parser.add_argument('--layer-start', type=int, default=0,
                        help='From which layer to start the exhaustive fault injection campaign')
    parser.add_argument('--layer-end', type=int, default=-1,
                        help='In which layer end the exhaustive fault injection campaign')
    parser.add_argument('--network', type=str, default='resnet20',
                        choices=['resnet20', 'densenet121', 'mobilenet-v2'])

    args = parser.parse_args()

    _layer_start = args.layer_start
    _layer_end = args.layer_end
    _network_name = args.network

    print(f'Running fault injection on {_network_name}')
    main(_layer_start, _layer_end, _network_name)
