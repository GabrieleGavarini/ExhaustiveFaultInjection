import os
import csv
import itertools
import numpy as np
from tqdm import tqdm

import torch
import torchmetrics

from BitFlipFI import BitFlipFI


def exhaustive_fault_injection(net: torch.nn.Module,
                               net_name: str,
                               net_layer_shape: list,
                               loader: torch.utils.data.DataLoader,
                               device: str,
                               y_golden: list,
                               iou_golden: list = None,
                               layer_start: int = 0,
                               layer_end: int = -1,
                               avoid_mantissa: bool = False,
                               mode: str = 'classification'):
    """
    Perform an exhaustive fault injection campaign on the target network
    :param net: The pytorch network on which to perform the fault injection campaign
    :param net_name: The network name
    :param net_layer_shape: A list containing the shape of each layer of the network
    :param loader: A pytorch DataLoader containing the loaded dataset
    :param device: the device where to run the fault injection campaign
    :param y_golden: A list containing the golden results of the network for the target dataset. If mode is 'classification'
    it contains the class predicted by the network. If mode is 'segmentation' it contains the golden segmentation mask
    :param iou_golden:A list containing the golden IoU of the network on the target dataset. Used only if mode is 'segmentation'.
    Default None
    :param layer_start: The first layer in which to perform a fault injection campaign. Default 0
    :param layer_end: The last layer in which to perform a fault injection campaign. Default -1
    :param avoid_mantissa: If True, faults are not injected in the mantissa bit. Default False
    :param mode: One of 'segmentation' or 'classification'. The task of the target network. Default 'segmentation'
    """

    with torch.set_grad_enabled(False):
        net.eval()

        # Set the output filename
        cwd = os.getcwd()
        os.makedirs(f'{cwd}/fault_injection/{net_name}', exist_ok=True)
        if layer_end == -1:
            filename = f'{cwd}/fault_injection/{net_name}/{layer_start}-{len(net_layer_shape) + layer_start}_exhaustive_results.csv'
        else:
            filename = f'{cwd}/fault_injection/{net_name}/{layer_start}-{layer_end}_exhaustive_results.csv'

        # Write header column for the output file
        with open(filename, 'w', newline='') as f_inj:
            writer_inj = csv.writer(f_inj)
            if mode == 'classification':
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
            elif mode == 'segmentation':
                writer_inj.writerow(['Injection',
                                     'Layer',
                                     'ImageIndex',
                                     'IoU_golden',
                                     'IoU_var',
                                     'ZeroMask',
                                     'Bit',
                                     'NoChange'])
            f_inj.flush()

            # If avoiding mantissa, inject only in the exponent and sign bits
            bit_start = 23 if avoid_mantissa else 0

            injection_index = 0

            # Count how many fault to inject
            total = np.sum(np.array([np.prod(layer_shape) for layer_shape in net_layer_shape])) * (32 - bit_start)

            # Start the exhaustive fault injection campaign
            pbar = tqdm(net_layer_shape, total=total)
            for layer, layer_shape in enumerate(pbar):
                for k in np.arange(layer_shape[0]):
                    for dim1 in np.arange(layer_shape[1]):
                        # Fault injection campaign for fully connected layers
                        if len(layer_shape) == 2:
                            for bit in np.arange(bit_start, 32):
                                run_fault_injection_inference(net=net,
                                                              pbar=pbar,
                                                              device=device,
                                                              loader=loader,
                                                              injection_index=injection_index,
                                                              writer_inj=writer_inj,
                                                              layer=layer,
                                                              layer_start=layer_start,
                                                              bit=bit,
                                                              k=k,
                                                              dim1=dim1,
                                                              y_golden=y_golden,
                                                              iou_golden=iou_golden,
                                                              mode=mode)
                                f_inj.flush()
                                injection_index += 1
                                torch.cuda.empty_cache()
                        # Fault injection campaign for convolutional layers
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
                                                                      layer=layer,
                                                                      layer_start=layer_start,
                                                                      bit=bit,
                                                                      k=k,
                                                                      dim1=dim1,
                                                                      dim2=dim2,
                                                                      dim3=dim3,
                                                                      y_golden=y_golden,
                                                                      iou_golden=iou_golden,
                                                                      mode=mode)
                                        f_inj.flush()
                                        injection_index += 1
                                        torch.cuda.empty_cache()


def run_golden_inference(net: torch.nn.Module,
                         net_name: str,
                         net_layer_shape: list,
                         loader: torch.utils.data.DataLoader,
                         device: str,
                         layer_start: int = 0,
                         layer_end: int = -1,
                         mode: str = 'classification'):
    """
    Run a golden (fault free) campaign of the target network on the target dataset
    :param net: The pytorch network on which to perform the fault injection campaign
    :param net_name: The network name
    :param net_layer_shape: A list containing the shape of each layer of the network
    :param loader: A pytorch DataLoader containing the loaded dataset
    :param device: the device where to run the fault injection campaign
    :param layer_start: The first layer in which to perform a fault injection campaign. Default 0
    :param layer_end: The last layer in which to perform a fault injection campaign. Default -1
    :param mode: One of 'segmentation' or 'classification'. The task of the target network. Default 'segmentation'
    :return: If mode is 'classification', returns a list containing the class predicted by the network for each image of
    the dataset. If mode is 'segmentation', returns a list containing the golden segmentation mask for each image of the
    dataset
    """
    correct = 0
    total = 0

    # General utils
    y_golden = []

    # Segmentation utils
    iou_golden = []
    iou = torchmetrics.IoU(num_classes=2,
                           reduction='none',
                           absent_score=0)

    # Create the output filename
    folder_name = f'./golden/{net_name}'
    os.makedirs(folder_name, exist_ok=True)
    if layer_end == -1:
        filename = f'{folder_name}/{layer_start}-{len(net_layer_shape) + layer_start}_golden.csv'
    else:
        filename = f'{folder_name}/{layer_start}-{layer_end}_golden.csv'

    with open(filename, 'w', newline='') as f_inj:
        writer_inj = csv.writer(f_inj)

        # Write the header in the output file
        if mode == 'classification':
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
        elif mode == 'segmentation':
            writer_inj.writerow(['Injection',
                                 'Layer',
                                 'ImageIndex',
                                 'IoU_golden',
                                 'IoU_var',
                                 'ZeroMask',
                                 'Bit',
                                 'NoChange'])

        # Begin the inference run
        pbar = tqdm(loader, desc='Golden Run')
        for image_index, data in enumerate(pbar):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            y_pred = net(x)

            # Classification task
            if mode == 'classification':
                # Compute the top5 prediction
                top_5 = torch.topk(y_pred, 5)

                # Compute the network accuracy
                pred = torch.topk(y_pred, 1)
                correct += np.sum([bool(pred.indices[i] == y_true[i]) for i in range(0, len(y_true))])
                total += len(y_true)
                accuracy = 100 * correct / total
                pbar.set_postfix({'Accuracy': accuracy})

                # Print the result of each image of the batch in the output file
                batch_length = len(top_5.indices)
                for index in range(0, batch_length):
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

                    y_golden += [int(top_5.indices[index][0])]

                del x, y_true, y_pred, top_5, pred
            elif mode == 'segmentation':
                batch_iou = iou(y_pred.cpu(), (y_true/max(1, y_true.max())).int().cpu())

                iou_golden += batch_iou
                y_golden += y_pred.detach()

                average_IoU = np.mean(iou_golden)

                pbar.set_postfix({'Average IoU': average_IoU})

                for index in range(0, len(batch_iou)):
                    output_list = [0,
                                   0,
                                   image_index * loader.batch_size + index,
                                   float(batch_iou[index]),
                                   0,
                                   float(batch_iou[index]) == 0,
                                   0,
                                   False]
                    writer_inj.writerow(output_list)
                    f_inj.flush()

    return torch.stack(y_golden), iou_golden


def run_fault_injection_inference(net: torch.nn.Module,
                                  loader: torch.utils.data.DataLoader,
                                  device: str,
                                  injection_index: int,
                                  writer_inj: csv.writer,
                                  layer: int,
                                  bit: int,
                                  k: int,
                                  dim1: int,
                                  dim2: int = None,
                                  dim3: int = None,
                                  y_golden: list = None,
                                  iou_golden: list = None,
                                  layer_start: int = 0,
                                  pbar: tqdm = None,
                                  mode='classification'):
    """
    Run a golden (fault free) campaign of the target network on the target dataset
    :param net: The pytorch network on which to perform the fault injection campaign
    :param loader: A pytorch DataLoader containing the loaded dataset
    :param device: the device where to run the fault injection campaign
    :param injection_index: The index of the current injection
    :param writer_inj: The instance of the csv writer where to output the results of the fault injection campaign
    :param layer: The index of the layer where to inject faults
    :param bit: The index of the bit where to inject the faults
    :param k: The index of the channel where to inject the faults
    :param dim1: The index of the first dimension where to inject the faults.
    :param dim2: The index of the second dimension where to inject the faults. Default None
    :param dim3: The index of the third dimension where to inject the faults. Default None
    :param y_golden: A list containing the golden results of the network for the target dataset. If mode is 'classification'
    it contains the class predicted by the network. If mode is 'segmentation' it contains the golden segmentation mask
    :param iou_golden:A list containing the golden IoU of the network on the target dataset. Used only if mode is 'segmentation'.
    Default None
    :param layer_start: The first layer in which to perform a fault injection campaign. Default 0
    :param pbar: a tqdm pbar to which append intermediate results of the fault injection campaign. Default None
    :param mode: One of 'segmentation' or 'classification'. The task of the target network. Default 'segmentation'
    """

    # Create the fault index for fully connected or convolutional
    if dim1 is None:
        fault = [layer + layer_start, k, dim1, [], [], bit]
    else:
        fault = [layer + layer_start, k, dim1, dim2, dim3, bit]
    if pbar is not None:
        pbar.set_description(f'fault: {fault}')

    # Segmentation utils
    iou_faulty = []
    iou = torchmetrics.IoU(num_classes=2,
                           reduction='none',
                           absent_score=0)

    # General Utils
    correct = 0
    total = 0

    # Inject the fault
    pfi_model = BitFlipFI(net,
                          fault_location=fault,
                          batch_size=1,
                          input_shape=[3, 32, 32],
                          layer_types=["all"],
                          use_cuda=(device == 'cuda'))
    corrupt_net = pfi_model.declare_weight_bit_flip()

    # Fault Injection
    for image_index, data in enumerate(loader):
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)

        y_pred = corrupt_net(x)

        # Classification Fault Injection
        if mode == 'classification':

            top_5 = torch.topk(y_pred, 5)

            pred = torch.topk(y_pred, 1)
            correct += np.sum([bool(pred.indices[i] == y_true[i]) for i in range(0, len(y_true))])
            total += len(y_true)
            accuracy = 100 * correct / total

            memory_reserved = torch.cuda.memory_reserved(0) / (1024*1024*1024)

            if pbar is not None:
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
                               int(y_golden[image_index * loader.batch_size + index]) if y_golden is not None else int(y_true[index]),  # int(y_golden.indices[index][0]),
                               bit,
                               False]
                writer_inj.writerow(output_list)

        # Segmentation Fault Injection
        elif mode == 'segmentation':
            batch_iou_golden = iou_golden[image_index * loader.batch_size: (image_index + 1) * loader.batch_size]
            batch_iou_faulty = iou(y_pred.cpu(), (y_true/max(1, y_true.max())).int().cpu())

            y_golden_batch = y_golden[image_index * loader.batch_size: (image_index + 1) * loader.batch_size].round().int()
            batch_iou = iou(y_pred.cpu(), y_golden_batch.cpu())

            iou_faulty += batch_iou_faulty.detach()
            average_IoU = np.mean(iou_faulty)

            if pbar is not None:
                pbar.set_postfix({'Average IoU': average_IoU})

            for index in range(0, len(batch_iou)):
                output_list = [0,
                               0,
                               image_index * loader.batch_size + index,
                               float(batch_iou_faulty[index]) / float(batch_iou_golden[index]) if float(batch_iou_golden[index]) else 1,
                               float(batch_iou[index]),
                               float(batch_iou_golden[index]) == 0,
                               bit,
                               False]
                writer_inj.writerow(output_list)

    if pbar is not None:
        pbar.update(1)
