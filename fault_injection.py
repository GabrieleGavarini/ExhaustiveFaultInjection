import os
import csv
import itertools
import numpy as np
from tqdm import tqdm

import torch
import torchmetrics

from BitFlipFI import BitFlipFI


def exhaustive_fault_injection(net,
                               net_name,
                               net_layer_shape,
                               loader,
                               device,
                               layer_start=0,
                               layer_end=-1,
                               avoid_mantissa=False,
                               y_golden=None,
                               iou_golden=None,
                               mode='classification'):

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
                                                              dim3=[],
                                                              y_golden=y_golden,
                                                              iou_golden=iou_golden,
                                                              mode=mode)
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
                                                                      dim3=dim3,
                                                                      y_golden=y_golden,
                                                                      iou_golden=iou_golden,
                                                                      mode=mode)
                                        injection_index += 1
                                        torch.cuda.empty_cache()


def run_golden_inference(loader, device, net, net_name, layer_start, layer_end, net_layer_shape, mode='classification'):
    correct = 0
    total = 0

    y_golden = []
    iou_golden = []
    iou = torchmetrics.IoU(num_classes=2,
                           reduction='none',
                           absent_score=0)

    folder_name = f'./golden/{net_name}'
    os.makedirs(folder_name, exist_ok=True)

    if layer_end == -1:
        filename = f'{folder_name}/{layer_start}-{len(net_layer_shape) + layer_start}_golden.csv'
    else:
        filename = f'{folder_name}/{layer_start}-{layer_end}_golden.csv'

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

        pbar = tqdm(loader, desc='Golden Run')
        for image_index, data in enumerate(pbar):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            y_pred = net(x)

            if mode == 'classification':
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


def run_fault_injection_inference(net, pbar, device, loader, injection_index, writer_inj, f_inj, layer, layer_start, bit, k, dim1=None, dim2=None, dim3=None, y_golden=None, iou_golden=None, mode='classification'):
    if dim1 is None:
        fault = [layer + layer_start, k, bit]
    else:
        fault = [layer + layer_start, k, dim1, dim2, dim3, bit]

    pbar.set_description(f'fault: {fault}')

    iou_faulty = []
    iou = torchmetrics.IoU(num_classes=2,
                           reduction='none',
                           absent_score=0)

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

        if mode == 'classification':

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
                               int(y_golden[image_index * loader.batch_size + index]) if y_golden is not None else int(y_true[index]),  # int(y_golden.indices[index][0]),
                               bit,
                               False]
                writer_inj.writerow(output_list)
                f_inj.flush()
        elif mode == 'segmentation':
            batch_iou_golden = iou_golden[image_index * loader.batch_size: (image_index + 1) * loader.batch_size]
            batch_iou_faulty = iou(y_pred.cpu(), (y_true/max(1, y_true.max())).int().cpu())

            y_golden_batch = y_golden[image_index * loader.batch_size: (image_index + 1) * loader.batch_size].round().int()
            batch_iou = iou(y_pred.cpu(), y_golden_batch.cpu())

            iou_faulty += batch_iou_faulty.detach()
            average_IoU = np.mean(iou_faulty)

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
                f_inj.flush()

    pbar.update(1)