import copy
import math

import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from matplotlib import pyplot as plt


# color_bar_critical_1 = '#5b8ba8'
# color_bar_critical_2 = '#1c506e'
# color_bar_n_1 = '#f3880a'
# color_bar_n_2 = '#f3880a'


def plot_graph(biased_sample_p_n,
               biased_sample_error,
               date_per_layer_p_n,
               date_per_layer_error,
               exhaustive_p,
               layer,
               mode='complete',
               number_biased_samples=10,
               number_per_layer_samples=10):

    exhaustive_color = '#b33939'
    color_bar_critical_1 = '#2c2c54'
    # color_bar_critical_2 = '#ffb142'
    color_bar_critical_2 = '#ff793f'
    # color_bar_n_1 = '#706fd3'
    color_bar_n_1 = '#33d9b2'
    if mode == 'injected':
        # color_bar_n_2 = '#34ace0'
        color_bar_n_2 = '#33d9b2'
    else:
        # color_bar_n_2 = '#706fd3'
        color_bar_n_2 = '#33d9b2'

    total_number_of_samples = number_biased_samples + number_per_layer_samples

    fig, ax1 = plt.subplots(1, figsize=(12, 6))

    critical_top = 10

    xticks = np.arange(0, total_number_of_samples)
    xticks_labels = [f'S{i}' for i in np.arange(1, number_biased_samples + 1)] + [f'S{i + number_biased_samples}' for i in np.arange(1, number_per_layer_samples + 1)]

    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks_labels, rotation=90)

    if mode == 'complete':
        width = .25
        x_1 = np.arange(0, number_biased_samples) - width / 2
        x_2 = x_1 + number_per_layer_samples
        x_3 = np.arange(0, number_biased_samples) + width / 2
        x_4 = x_3 + number_per_layer_samples
        ax1.set_xlim(0 - 2 * width, total_number_of_samples - 2 * width)
        ax2 = ax1.twinx()
    else:
        width = .5
        x_1 = np.arange(0, number_biased_samples)
        x_2 = x_1 + number_per_layer_samples
        x_3 = x_1
        x_4 = x_2
        ax1.set_xlim(0 - width, total_number_of_samples - width)
        ax2 = ax1

    if mode == 'complete' or mode == 'faults':
        ax1.bar(x=x_1,
                height=[p * 100 for p, n in biased_sample_p_n],
                yerr=[error * 100 for error in biased_sample_error],
                error_kw=dict(ecolor='grey', alpha=.95, lw=1.25, capsize=2, capthick=.5, zorder=2),
                width=width,
                color=color_bar_critical_1,
                label='Proposed (left)',
                zorder=1)
        ax1.bar(x=x_2,
                height=[p * 100 for p, n in date_per_layer_p_n],
                yerr=[error * 100 for error in date_per_layer_error],
                error_kw=dict(ecolor='black', alpha=.75, lw=1.25, capsize=2, capthick=.5, zorder=2),
                width=width,
                color=color_bar_critical_2,
                label='DATE09 per layer (left)',
                zorder=1)

        ax1.axhline(exhaustive_p, linestyle='--', linewidth=1, color=exhaustive_color, zorder=99999,
                    label='Exhaustive FI results (left)')

        yticks = [i for i in np.linspace(0, critical_top, 6)]
        yticks.append(exhaustive_p)
        yticks_labels = [f'{i:.2f}' for i in yticks]
        ax1.set_ylim(0, critical_top)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(yticks_labels)
        ax1.set_ylabel('Critical Faults [%]')
        ax1.get_yticklabels()[-1].set_color(exhaustive_color)

        if mode == 'complete':
            ax1.bar(x=0,
                    height=0,
                    width=0,
                    color=color_bar_n_1,
                    label='Injected Faults (right)')

        ax1.legend(loc='upper left')

    if mode == 'complete' or mode == 'injected':
        ax2.bar(x=x_3,
                height=[n / 10000 for p, n in biased_sample_p_n],
                width=width,
                color=color_bar_n_1,
                label='Proposed')
        ax2.bar(x=x_4,
                height=[n / 10000 for p, n in date_per_layer_p_n],
                width=width,
                color=color_bar_n_2,
                label='DATE09 per layer')

        yticks = [i for i in np.linspace(0, 1.5, 7)]
        yticks_labels = [f'{int(i * 10000): ,}' for i in yticks]
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(yticks_labels)
        ax2.set_ylabel('Injected Faults (n)')

    if mode == 'injected':
        ax2.legend(loc='upper left')

    fig.show()
    os.makedirs('plot/DATEvsProposed', exist_ok=True)
    fig.savefig(f'plot/DATEvsProposed/{str(layer).zfill(2)}_{mode}_critical.png')


def poisson_variance(p, n):
    return n * p


def normal_variance(p, n):
    return n * p * (1 - p)


def finite_population_correction(n, N):
    return (N - n) / (n * (N - 1))


def compute_error_margin(p, n, N=None, t=2.58, finite_population=True):
    """                                                                                                         .
00    .     Compute the error margin for a given p, n, N and t. It is possible to specify a finite_population factor
    :param p: The sampled p
    :param n: The number of samples
    :param N: Default None. The population size. Only used if finite_population is True
    :param t: The z-index of the confidence
    :param finite_population: Default True. Whether to use the finite population correction factor
    :return: the error margin
    """

    population_variance = normal_variance(p, n)

    if finite_population:
        # sample_variance = population_variance * finite population factor
        sample_variance = population_variance * finite_population_correction(n, N)
    else:
        # sample_variance = population_variance / n
        sample_variance = population_variance / n

    # The margin of error
    return t * math.sqrt(sample_variance / n)


def return_complete_df(image_df):

    result_dicts = []
    for bit in np.arange(0, 32):
        complete_df_bit = copy.deepcopy(image_df[image_df.Bit == bit])

        # We are actually doing bit-flips since all the stuck-at that do not change value are always masked
        adjusted_length = len(complete_df_bit)

        masked = np.sum(complete_df_bit.Top_1 == complete_df_bit.Golden) / adjusted_length
        critical = 1 - masked

        result_dicts.append({'Bit': bit,
                             'Masked': masked * 100,
                             'Critical': critical * 100,
                             'InjectedFaults': adjusted_length})
    return result_dicts


def _create_sample_per_layer(df, n, random_state=None):
    return df.sample(n, axis=0, random_state=random_state)


def create_sample(df, n, random_state=None):
    if isinstance(n, int):
        return _create_sample_per_layer(df, n, random_state)

    sample_df_list = []

    for bit in np.arange(0, 32):
        sample_df_list.append(df[df.Bit == bit].sample(n=int(n[bit]), axis=0, random_state=random_state))

    return pd.concat(sample_df_list)


def return_p_and_n(df, convert_p_to_probability=True):
    p = df.Critical.mean() / 100 if convert_p_to_probability else df.Critical.mean()
    n = df.InjectedFaults.sum()

    return p, n


def extract_samples_p_n(layer,
                        net_name,
                        number_biased_samples,
                        number_unbiased_samples,
                        number_per_layer_samples,
                        number_of_samples_folder='../fault_injection/number_of_samples',
                        load_if_exist=True,
                        load_folder='intermediate_results',
                        seed=1234):

    # load N from file
    number_of_samples_folder = f'{number_of_samples_folder}/{net_name}'
    N = pd.read_csv(f'{number_of_samples_folder}/N_layers.csv', index_col=0).loc[layer]['N']

    try:
        if not load_if_exist:
            raise FileNotFoundError

        biased_sample_p_n = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_biased_samples}_biased_sample_p_n.npy')
        biased_sample_error = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_biased_samples}_biased_sample_error.npy')

        unbiased_sample_p_n = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_unbiased_samples}_unbiased_sample_p_n.npy')
        unbiased_sample_error = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_unbiased_samples}_unbiased_sample_error.npy')

        date_per_layer_p_n = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_per_layer_samples}_date_per_layer_p_n.npy')
        date_per_layer_error = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_per_layer_samples}_date_per_layer_error.npy')

        exhaustive_p = float(np.load(f'{load_folder}/{str(layer).zfill(2)}/exhaustive_p.npy'))

        print('Loaded from saved files...')

    except FileNotFoundError:

        os.makedirs(f'{load_folder}/{str(layer).zfill(2)}', exist_ok=True)

        # ---- BEGIN DATAFRAME CREATION ---- #
        image_merge_folder = f'../fault_injection/resnet20/'

        df_list = []
        for filename in os.listdir(image_merge_folder):
            df_list.append(pd.read_csv(f'{image_merge_folder}/{filename}', low_memory=False).dropna())
        image_df = pd.concat(df_list, ignore_index=True)
        image_df = image_df.drop(['Top_2', 'Top_3', 'Top_4', 'Top_5'], axis=1)

        image_df = image_df[image_df.Layer.astype(int) == layer]

        stuck_at_df = copy.deepcopy(image_df)
        stuck_at_df.Top_1 = stuck_at_df.Golden
        stuck_at_df.Injection = stuck_at_df.Injection + image_df.Injection.max() + 1
        stuck_at_df.NoChange = True

        image_df = pd.concat([image_df, stuck_at_df])

        # ---- END DATAFRAME CREATION ---- #

        random_state = np.random.default_rng(seed=seed).bit_generator

        # Biased with critical
        biased_n_by_bit = pd.read_csv(f'{number_of_samples_folder}/n_tuning_scenario_1.csv', index_col=0).loc[layer].to_list()
        biased_n_by_bit.reverse()

        biased_sample_list = [
            pd.DataFrame(return_complete_df(create_sample(image_df,
                                                          n=biased_n_by_bit,
                                                          random_state=random_state))).set_index('Bit')
            for _ in tqdm(np.arange(0, number_biased_samples), desc='Sampling biased')]
        biased_sample_p_n = [return_p_and_n(biased_sample) for biased_sample in biased_sample_list]
        biased_sample_error = [compute_error_margin(p, n, N, finite_population=True) for p, n in biased_sample_p_n]
        np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_biased_samples}_biased_sample_p_n.npy',  np.asarray(biased_sample_p_n))
        np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_biased_samples}_biased_sample_error.npy',  np.asarray(biased_sample_error))

        # Unbiased
        unbiased_n_by_bit = pd.read_csv(f'{number_of_samples_folder}/n_worst_case.csv', index_col=0).loc[layer].to_list()
        unbiased_n_by_bit.reverse()

        unbiased_sample_list = [
            pd.DataFrame(return_complete_df(create_sample(image_df,
                                                          n=unbiased_n_by_bit,
                                                          random_state=random_state))).set_index('Bit')
            for _ in tqdm(np.arange(0, number_unbiased_samples), desc='Sampling unbiased')]
        unbiased_sample_p_n = [return_p_and_n(unbiased_sample) for unbiased_sample in unbiased_sample_list]
        unbiased_sample_error = [compute_error_margin(p, n, N) for p, n in unbiased_sample_p_n]
        np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_unbiased_samples}_unbiased_sample_p_n.npy',  np.asarray(unbiased_sample_p_n))
        np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_unbiased_samples}_unbiased_sample_error.npy',  np.asarray(unbiased_sample_error))

        # Date per layer
        date_per_layer_n = int(pd.read_csv(f'{number_of_samples_folder}/n_date_per_layer.csv', index_col=0, header=None).loc[layer].values[0])

        date_per_layer_list = [
            pd.DataFrame(return_complete_df(create_sample(image_df,
                                                          n=date_per_layer_n,
                                                          random_state=random_state))).set_index('Bit')
            for _ in tqdm(np.arange(0, number_per_layer_samples), desc='Sampling DATE09 per layer')]
        date_per_layer_p_n = [return_p_and_n(date_per_layer) for date_per_layer in date_per_layer_list]
        date_per_layer_error = [compute_error_margin(p, n, N) for p, n in date_per_layer_p_n]
        np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_per_layer_samples}_date_per_layer_p_n.npy',  np.asarray(date_per_layer_p_n))
        np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_per_layer_samples}_date_per_layer_error.npy',  np.asarray(date_per_layer_error))

        # Exhaustive results
        print('Extracting exhaustive p...')
        results_dataframe_complete = pd.DataFrame(return_complete_df(image_df))
        results_dataframe_complete = results_dataframe_complete.set_index('Bit')
        exhaustive_p = results_dataframe_complete.Critical.mean()
        np.save(f'{load_folder}/{str(layer).zfill(2)}/exhaustive_p.npy',  np.asarray(exhaustive_p))

    sample_list = {}
    sample_list.update({f'Proposed sample {i}': [s[0][1], s[0][0], s[1]] for i, s in
                        enumerate(list(zip(*[biased_sample_p_n, biased_sample_error])))})
    sample_list.update({f'Unbiased sample {i}': [s[0][1], s[0][0], s[1]] for i, s in
                        enumerate(list(zip(*[unbiased_sample_p_n, unbiased_sample_error])))})
    sample_list.update({f'DATE209 per layer sample {i}': [s[0][1], s[0][0], s[1]] for i, s in
                        enumerate(list(zip(*[date_per_layer_p_n, date_per_layer_error])))})
    sample_list.update({'Exhaustive': [N, exhaustive_p, 0]})

    sample_df = pd.DataFrame(sample_list).transpose()
    sample_df.columns = ['n', 'p', 'error_margin']
    os.makedirs('csv', exist_ok=True)
    sample_df.to_csv(f'csv/{str(layer).zfill(2)}_samples.csv')

    return [biased_sample_p_n, biased_sample_error], [unbiased_sample_p_n, unbiased_sample_error], [date_per_layer_p_n, date_per_layer_error], exhaustive_p
