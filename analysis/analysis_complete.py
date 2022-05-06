from utils import extract_samples_p_n
from graph import Plotter
import numpy as np
import pandas as pd


total_layer_number = 52
layer_start = 0
layer_end = 7
avoid_mantissa=True

number_biased_samples = 10
number_unbiased_samples = 10
number_date_per_layer_samples = 10
number_date_samples = 10

net_name = 'mobilenet-v2'

biased_sample, unbiased_sample, date_per_layer,  date, exhaustive_p = extract_samples_p_n(layer_start=layer_start,
                                                                                          layer_end=layer_end,
                                                                                          net_name=net_name,
                                                                                          number_biased_samples=number_biased_samples,
                                                                                          number_unbiased_samples=number_unbiased_samples,
                                                                                          number_date_per_layer_samples=number_date_per_layer_samples,
                                                                                          number_date_samples=number_date_samples,
                                                                                          seed=51195,
                                                                                          load_if_exist=True,
                                                                                          avoid_mantissa=avoid_mantissa)

entry_list = []
for i in np.arange(layer_start, layer_end):
    for j in np.arange(0, 10):
        entry = {'Layer': i,
                 'Sample_Index': j,
                 'Exhaustive_p': exhaustive_p[i].mean()/100,
                 'data_aware_p': biased_sample[0][i][j][0],
                 'data_aware_error': biased_sample[1][i][j],
                 'data_aware_n': biased_sample[0][i][j][1],
                 'data_unaware_p': unbiased_sample[0][i][j][0],
                 'data_unaware_error': unbiased_sample[1][i][j],
                 'data_unaware_n': unbiased_sample[0][i][j][1],
                 'date_per_layer_p': date_per_layer[0][i][j][0],
                 'date_per_layer_error': date_per_layer[1][i][j],
                 'date_per_layer_n': date_per_layer[0][i][j][1],
                 'date_p': date[0][i][j][0],
                 'date_error': date[1][i][j],
                 'date_n': date[0][i][j][1]}
        entry_list.append(entry)

result_df = pd.DataFrame(entry_list)
result_df.to_csv(f'csv/{net_name}_complete_p_n.csv')

plotter = Plotter()
# plotter.plot_layers(biased_sample_p_n=biased_sample[0],
#                     unbiased_sample_p_n=unbiased_sample[0],
#                     date_per_layer_p_n=date_per_layer[0],
#                     date_p_n=date[0],
#                     exhaustive_p=np.mean(exhaustive_p, axis=1))

for layer in np.arange(layer_start, layer_end):
    plotter.plot_graph(biased_sample_p_n=biased_sample[0][layer],
                       biased_sample_error=biased_sample[1][layer],
                       date_per_layer_p_n=date_per_layer[0][layer],
                       date_per_layer_error=date_per_layer[1][layer],
                       exhaustive_p=np.mean(exhaustive_p[layer]),
                       layer=layer,
                       mode='faults',
                       number_biased_samples=number_biased_samples,
                       number_per_layer_samples=number_date_per_layer_samples,
                       critical_top=2)
