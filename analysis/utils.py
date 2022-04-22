import copy
import math

import pandas as pd
import numpy as np
import os

from tqdm import tqdm

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


def extract_samples_p_n(total_layer_number,
                        net_name,
                        number_biased_samples,
                        number_unbiased_samples,
                        number_date_per_layer_samples,
                        number_date_samples,
                        number_of_samples_folder='../fault_injection/number_of_samples',
                        load_if_exist=True,
                        load_folder='intermediate_results',
                        seed=1234):

    # load N from file
    number_of_samples_folder = f'{number_of_samples_folder}/{net_name}'
    N = pd.read_csv(f'{number_of_samples_folder}/N_layers.csv', index_col=0)['N']

    biased_sample_p_n = []
    biased_sample_error = []

    unbiased_sample_p_n = []
    unbiased_sample_error = []

    date_per_layer_p_n = []
    date_per_layer_error = []

    date_p_n = []
    date_error = []

    exhaustive_p = []

    try:
        if not load_if_exist:
            raise FileNotFoundError

        for layer in np.arange(0, total_layer_number):
            biased_sample_p_n.append(np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_biased_samples}_biased_sample_p_n.npy'))
            biased_sample_error.append(np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_biased_samples}_biased_sample_error.npy'))

            unbiased_sample_p_n.append(np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_unbiased_samples}_unbiased_sample_p_n.npy'))
            unbiased_sample_error.append(np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_unbiased_samples}_unbiased_sample_error.npy'))

            date_per_layer_p_n.append(np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_per_layer_samples}_date_per_layer_p_n.npy'))
            date_per_layer_error.append(np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_per_layer_samples}_date_per_layer_error.npy'))

            date_p_n.append(np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_samples}_date_p_n.npy'))
            date_error.append(np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_samples}_date_error.npy'))

            exhaustive_p.append(np.load(f'{load_folder}/{str(layer).zfill(2)}/exhaustive_p.npy'))

        print('Loaded from saved files...')

    except FileNotFoundError as not_found:

        print(f'File {not_found.filename} not found, computing results...')

        number_of_chunks = 1000
        chunksize = int((100 * N.values.sum()/2) / number_of_chunks)

        # ---- BEGIN DATAFRAME CREATION ---- #
        golden_filename = f'../golden/{net_name}/golden.csv'
        golden_df = pd.read_csv(golden_filename)

        image_merge_folder = f'../fault_injection/resnet20/'

        df_list = []
        with tqdm(total=number_of_chunks) as pbar:
            for filename in os.listdir(image_merge_folder):
                for chunk in pd.read_csv(f'{image_merge_folder}/{filename}', chunksize=chunksize):
                    pbar.set_description(f'Loading {filename}')
                    chunk = chunk.dropna()
                    if 'Golden' in chunk.columns:
                        chunk = chunk.drop('Golden', axis=1).merge(golden_df[['ImageIndex', 'Golden']], how='left', on='ImageIndex')
                    chunk = chunk[['Bit', 'Layer', 'Top_1', 'Golden']]
                    df_list.append(chunk)

                    stuck_at_chunk = copy.deepcopy(chunk)
                    stuck_at_chunk.Top_1 = stuck_at_chunk.Golden
                    df_list.append(stuck_at_chunk)

                    pbar.update(1)

        image_df = pd.concat(df_list, ignore_index=True)

        random_state = np.random.default_rng(seed=seed).bit_generator

        # ---- END DATAFRAME CREATION ---- #

        for layer in np.arange(0, total_layer_number):
            print(f'#### Layer {layer} ####')
            os.makedirs(f'{load_folder}/{str(layer).zfill(2)}', exist_ok=True)

            image_df_layer = image_df[image_df.Layer.astype(int) == layer]

            # # Biased with critical
            # biased_n_by_bit = pd.read_csv(f'{number_of_samples_folder}/n_tuning_scenario_1.csv', index_col=0).loc[layer].to_list()
            # biased_n_by_bit.reverse()
            #
            # biased_sample_list = [
            #     pd.DataFrame(return_complete_df(create_sample(image_df_layer,
            #                                                   n=biased_n_by_bit,
            #                                                   random_state=random_state))).set_index('Bit')
            #     for _ in tqdm(np.arange(0, number_biased_samples), desc='Sampling biased')]
            # biased_sample_p_n.append([return_p_and_n(biased_sample) for biased_sample in biased_sample_list])
            # biased_sample_error.append([compute_error_margin(p, n, N[layer], finite_population=True) for p, n in biased_sample_p_n[layer]])
            # np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_biased_samples}_biased_sample_p_n.npy',  np.asarray(biased_sample_p_n[layer]))
            # np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_biased_samples}_biased_sample_error.npy',  np.asarray(biased_sample_error[layer]))
            #
            # # Unbiased
            # unbiased_n_by_bit = pd.read_csv(f'{number_of_samples_folder}/n_worst_case.csv', index_col=0).loc[layer].to_list()
            # unbiased_n_by_bit.reverse()
            #
            # unbiased_sample_list = [
            #     pd.DataFrame(return_complete_df(create_sample(image_df_layer,
            #                                                   n=unbiased_n_by_bit,
            #                                                   random_state=random_state))).set_index('Bit')
            #     for _ in tqdm(np.arange(0, number_unbiased_samples), desc='Sampling unbiased')]
            # unbiased_sample_p_n.append([return_p_and_n(unbiased_sample) for unbiased_sample in unbiased_sample_list])
            # unbiased_sample_error.append([compute_error_margin(p, n, N[layer]) for p, n in unbiased_sample_p_n[layer]])
            # np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_unbiased_samples}_unbiased_sample_p_n.npy',  np.asarray(unbiased_sample_p_n[layer]))
            # np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_unbiased_samples}_unbiased_sample_error.npy',  np.asarray(unbiased_sample_error[layer]))
            #
            # # Date per layer
            # date_per_layer_n = int(pd.read_csv(f'{number_of_samples_folder}/n_date_per_layer.csv', index_col=0, header=None).loc[layer].values[0])
            #
            # date_per_layer_list = [
            #     pd.DataFrame(return_complete_df(create_sample(image_df_layer,
            #                                                   n=date_per_layer_n,
            #                                                   random_state=random_state))).set_index('Bit')
            #     for _ in tqdm(np.arange(0, number_date_per_layer_samples), desc='Sampling DATE09 per layer')]
            # date_per_layer_p_n.append([return_p_and_n(date_per_layer) for date_per_layer in date_per_layer_list])
            # date_per_layer_error.append([compute_error_margin(p, n, N[layer]) for p, n in date_per_layer_p_n[layer]])
            # np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_per_layer_samples}_date_per_layer_p_n.npy', np.asarray(date_per_layer_p_n[layer]))
            # np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_per_layer_samples}_date_per_layer_error.npy', np.asarray(date_per_layer_error[layer]))

            # Date
            date_n = int(pd.read_csv(f'{number_of_samples_folder}/n_date.csv', index_col=0, header=None).loc[layer].values[0])

            date_list = [
                pd.DataFrame(return_complete_df(create_sample(image_df_layer,
                                                              n=date_n,
                                                              random_state=random_state))).set_index('Bit')
                for _ in tqdm(np.arange(0, number_date_samples), desc='Sampling DATE09')]
            date_p_n.append([return_p_and_n(date) for date in date_list])
            date_error.append([compute_error_margin(p, n, N[layer]) for p, n in date_p_n[layer]])
            np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_samples}_date_p_n.npy', np.asarray(date_p_n[layer]))
            np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_samples}_date_error.npy', np.asarray(date_error[layer]))

            # Exhaustive results
            print('Extracting exhaustive p...')
            results_dataframe_complete = pd.DataFrame(return_complete_df(image_df_layer))
            results_dataframe_complete = results_dataframe_complete.set_index('Bit')
            exhaustive_p.append(results_dataframe_complete.Critical.values)
            np.save(f'{load_folder}/{str(layer).zfill(2)}/exhaustive_p.npy',  np.asarray(exhaustive_p[layer]))

    sample_list = {}
    sample_list.update({f'Proposed sample {i}': [s[0][1], s[0][0], s[1]] for i, s in
                        enumerate(list(zip(*[biased_sample_p_n, biased_sample_error])))})
    sample_list.update({f'Unbiased sample {i}': [s[0][1], s[0][0], s[1]] for i, s in
                        enumerate(list(zip(*[unbiased_sample_p_n, unbiased_sample_error])))})
    sample_list.update({f'DATE209 per layer sample {i}': [s[0][1], s[0][0], s[1]] for i, s in
                        enumerate(list(zip(*[date_per_layer_p_n, date_per_layer_error])))})
    sample_list.update({f'DATE209 sample {i}': [s[0][1], s[0][0], s[1]] for i, s in
                        enumerate(list(zip(*[date_p_n, date_error])))})
    sample_list.update({'Exhaustive': [N[layer], exhaustive_p, 0]})

    sample_df = pd.DataFrame(sample_list).transpose()
    sample_df.columns = ['n', 'p', 'error_margin']
    os.makedirs('csv', exist_ok=True)
    sample_df.to_csv(f'csv/{str(layer).zfill(2)}_samples.csv')

    return [biased_sample_p_n, biased_sample_error], [unbiased_sample_p_n, unbiased_sample_error], [date_per_layer_p_n, date_per_layer_error], [date_p_n, date_error], exhaustive_p
