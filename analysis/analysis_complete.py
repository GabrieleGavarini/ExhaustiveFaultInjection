from utils import extract_samples_p_n
from graph import Plotter
import numpy as np
import pandas as pd

total_layer_number = 19
number_biased_samples = 10
number_unbiased_samples = 10
number_date_per_layer_samples = 10
number_date_samples = 10

biased_sample, unbiased_sample, date_per_layer,  date, exhaustive_p = extract_samples_p_n(total_layer_number=total_layer_number,
                                                                                          net_name='resnet20',
                                                                                          number_biased_samples=number_biased_samples,
                                                                                          number_unbiased_samples=number_unbiased_samples,
                                                                                          number_date_per_layer_samples=number_date_per_layer_samples,
                                                                                          number_date_samples=number_date_samples,
                                                                                          seed=51195,
                                                                                          load_if_exist=True)

entry_list = []
for i in np.arange(0, total_layer_number):
    for j in np.arange(0, 10):
        entry = {'Layer': i,
                 'Sample_Index': j,
                 'Exhaustive_p': exhaustive_p[i].mean(),
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
result_df.to_csv('csv/complete_p_n.csv')

plotter = Plotter()
plotter.plot_layers(biased_sample_p_n=biased_sample[0],
                    unbiased_sample_p_n=unbiased_sample[0],
                    date_per_layer_p_n=date_per_layer[0],
                    exhaustive_p=np.mean(exhaustive_p, axis=1))

for layer in np.arange(0, total_layer_number):
    plotter.plot_graph(biased_sample_p_n=biased_sample[0][layer],
                       biased_sample_error=biased_sample[1][layer],
                       date_per_layer_p_n=date_per_layer[0][layer],
                       date_per_layer_error=date_per_layer[1][layer],
                       exhaustive_p=np.mean(exhaustive_p, axis=1)[layer],
                       layer=layer,
                       mode='faults',
                       number_biased_samples=number_biased_samples,
                       number_per_layer_samples=number_date_per_layer_samples)
