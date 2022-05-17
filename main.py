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


def run_golden_inference(loader, device, net, layer_start, layer_end, net_layer_shape):
    correct = 0
    total = 0

    folder_name = './golden/mobilenet-v2'
    os.makedirs(folder_name, exist_ok=True)

    if layer_end == -1:
        filename = f'{folder_name}/{layer_start}-{len(net_layer_shape) + layer_start}_golden.csv'
    else:
        filename = f'{folder_name}/{layer_start}-{layer_end}_golden.csv'

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

        pbar = tqdm(loader, desc='Golden Run')
        for image_index, data in enumerate(pbar):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            y_pred = net(x)

            top_5 = torch.topk(y_pred, 5)

            pred = torch.topk(y_pred, 1)
            correct += np.sum([bool(pred.indices[i] == y_true[i]) for i in range(0, len(y_true))])
            total += len(y_true)
            accuracy = 100 * correct / total

            pbar.set_postfix({'Accuracy': accuracy})

            for index in range(0, len(top_5.indices)):
                output_list = [0,
                               0,
                               image_index * loader.batch_size + index,
                               int(top_5.indices[index][0]),
                               int(top_5.indices[index][1]),
                               int(top_5.indices[index][2]),
                               int(top_5.indices[index][3]),
                               int(top_5.indices[index][4]),
                               int(y_true[index]),
                               0,
                               False]
                writer_inj.writerow(output_list)
                f_inj.flush()

            del x, y_true, y_pred, top_5, pred


def run_fault_injection_inference(net, pbar, device, loader, injection_index, writer_inj, f_inj, layer, layer_start, bit, k, dim1=None, dim2=None, dim3=None):
    if dim1 is None:
        fault = [layer + layer_start, k, bit]
    else:
        fault = [layer + layer_start, k, dim1, dim2, dim3, bit]

    pbar.set_description(f'fault: {fault}')

    pfi_model = BitFlipFI(net,
                          fault_location=fault,
                          batch_size=1,
                          input_shape=[3, 32, 32],
                          layer_types=["all"],
                          use_cuda=(device == 'cuda'))

    corrupt_net = pfi_model.declare_weight_bit_flip()
    correct = 0
    total = 0

    for image_index, data in enumerate(loader):
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)

        y_pred = corrupt_net(x)

        top_5 = torch.topk(y_pred, 5)

        pred = torch.topk(y_pred, 1)
        correct += np.sum([bool(pred.indices[i] == y_true[i]) for i in range(0, len(y_true))])
        total += len(y_true)
        accuracy = 100 * correct / total

        memory_reserved = torch.cuda.memory_reserved(0) / (1024*1024*1024)
        pbar.set_postfix({'Accuracy': accuracy, 'Reserved GB': memory_reserved})

        for index in range(0, len(top_5.indices)):
            output_list = [injection_index,
                           layer + layer_start,
                           image_index * loader.batch_size + index,
                           int(top_5.indices[index][0]),
                           int(top_5.indices[index][1]),
                           int(top_5.indices[index][2]),
                           int(top_5.indices[index][3]),
                           int(top_5.indices[index][4]),
                           int(y_true[index]),  # int(y_golden.indices[index][0]),
                           bit,
                           False]
            writer_inj.writerow(output_list)
            f_inj.flush()

    pbar.update(1)


def main(layer_start=0,
         layer_end=-1,
         network_name='resnet20',
         test_image_per_class=1,
         avoid_mantissa=False,
         target_memory_GB=None,
         batch_size=10,
         use_cuda=True):

    device = 'cpu' if (not use_cuda) or (not torch.cuda.is_available()) else 'cuda'
    torch.device(device)
    print(f'Running on device {device}')

    if avoid_mantissa:
        print('Avoiding injection fault on the mantissa')

    # Limit GPU memory
    if (device == 'cuda') and (target_memory_GB is not None):
        total_memory = torch.cuda.get_device_properties(0).total_memory
        target_memory = target_memory_GB * (1024*1024*1024)
        memory_fraction = target_memory / total_memory
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=None)

    _, _, test_loader = load_CIFAR10_datasets(test_batch_size=batch_size, test_image_per_class=test_image_per_class)

    if network_name == 'resnet20':
        network = resnet20()
        network_path = 'models/pretrained_models/resnet20-trained.th'
    elif network_name == 'densenet121':
        network = densenet121()
        network_path = 'models/pretrained_models/densenet121.pt'
    elif network_name == 'mobilenet-v2':
        network = MobileNetV2()
        network_path = 'models/pretrained_models/mobilenet.pth'

    network.to(device)
    load_from_dict(network=network,
                   device=device,
                   path=network_path)

    network_layers = []
    for name, param in network.named_parameters():
        if "weight" in name and ("features" in name or "conv" in name or "linear" in name):
            network_layers.append(param)

    if layer_end == -1:
        network_layers = network_layers[layer_start:]
    else:
        network_layers = network_layers[layer_start:layer_end]

    network_layers_shape = [layer.shape for layer in network_layers]

    run_golden_inference(loader=test_loader,
                         device=device,
                         net=network,
                         layer_start=layer_start,
                         layer_end=layer_end,
                         net_layer_shape=network_layers_shape)
    torch.cuda.empty_cache()

    exhaustive_fault_injection(net=network,
                               net_name=network_name,
                               net_layer_shape=network_layers_shape,
                               loader=test_loader,
                               device=device,
                               layer_start=layer_start,
                               layer_end=layer_end,
                               avoid_mantissa=avoid_mantissa)


def exhaustive_fault_injection(net,
                               net_name,
                               net_layer_shape,
                               loader,
                               device,
                               layer_start=0,
                               layer_end=-1,
                               avoid_mantissa=False):

    cwd = os.getcwd()
    os.makedirs(f'{cwd}/fault_injection/{net_name}', exist_ok=True)

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

            bit_start = 23 if avoid_mantissa else 0

            injection_index = 0
            total = np.sum(np.array([np.prod(layer_shape) for layer_shape in net_layer_shape])) * (32 - bit_start)
            pbar = tqdm(net_layer_shape, total=total)
            for layer, layer_shape in enumerate(pbar):
                for k in np.arange(layer_shape[0]):
                    for dim1 in np.arange(layer_shape[1]):
                        if len(layer_shape) == 2:
                            for bit in np.arange(bit_start, 32):
                                run_fault_injection_inference(net=net,
                                                              pbar=pbar,
                                                              device=device,
                                                              loader=loader,
                                                              injection_index=injection_index,
                                                              writer_inj=writer_inj,
                                                              f_inj=f_inj,
                                                              layer=layer,
                                                              layer_start=layer_start,
                                                              bit=bit,
                                                              k=k,
                                                              dim1=dim1,
                                                              dim2=[],
                                                              dim3=[])
                                injection_index += 1
                                torch.cuda.empty_cache()
                        else:
                            for dim2 in np.arange(layer_shape[2]):
                                for dim3 in np.arange(layer_shape[3]):
                                    for bit in np.arange(bit_start, 32):
                                        run_fault_injection_inference(net=net,
                                                                      pbar=pbar,
                                                                      device=device,
                                                                      loader=loader,
                                                                      injection_index=injection_index,
                                                                      writer_inj=writer_inj,
                                                                      f_inj=f_inj,
                                                                      layer=layer,
                                                                      layer_start=layer_start,
                                                                      bit=bit,
                                                                      k=k,
                                                                      dim1=dim1,
                                                                      dim2=dim2,
                                                                      dim3=dim3)
                                        injection_index += 1
                                        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exhaustive fault injection')
    parser.add_argument('--layer-start', type=int, default=0,
                        help='From which layer to start the exhaustive fault injection campaign')
    parser.add_argument('--layer-end', type=int, default=-1,
                        help='In which layer end the exhaustive fault injection campaign')
    parser.add_argument('--network', type=str, default='resnet20',
                        choices=['resnet20', 'densenet121', 'mobilenet-v2'])
    parser.add_argument('--image_per_class', type=int, default=1,
                        help='How many image per class for each inference run')
    parser.add_argument('--avoid_mantissa', action='store_true', default=False,
                        help='Whether or not to inject faults in the mantissa')
    parser.add_argument('--target_memory_GB', type=float, default=1,
                        help='How many GigaByte of GPU memory the process is allowed to use')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for the inference')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Whether to use cuda or not if they are available')

    args = parser.parse_args()

    _layer_start = args.layer_start
    _layer_end = args.layer_end
    _network_name = args.network
    _avoid_mantissa = args.avoid_mantissa
    _image_per_class = args.image_per_class
    _target_memory_GB = args.target_memory_GB
    _batch_size = args.batch_size
    _use_cuda = args.use_cuda

    print(f'Running fault injection on {_network_name}')
    main(layer_start=_layer_start,
         layer_end=_layer_end,
         network_name=_network_name,
         test_image_per_class=_image_per_class,
         avoid_mantissa=_avoid_mantissa,
         target_memory_GB=_target_memory_GB,
         batch_size=_batch_size,
         use_cuda=_use_cuda)
