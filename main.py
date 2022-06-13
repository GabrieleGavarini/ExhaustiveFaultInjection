import torch

import argparse

from fault_injection import exhaustive_fault_injection, run_golden_inference
from models.utils import load_CIFAR10_datasets, load_from_dict
from models.resnet import resnet20
from models.densenet import densenet121
from models.mobilenetv2 import MobileNetV2


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

    y_golden = run_golden_inference(loader=test_loader,
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
                               avoid_mantissa=avoid_mantissa,
                               y_golden=y_golden)


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
