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


def _create_df(net_name, chunksize, number_of_chunks, layer=None):
    """
    Create a dataframe with all the bit-flip faults and their effect
    :param net_name: The name of the network
    :param chunksize: The size of the chunks to load
    :param number_of_chunks: Then umber of chunks
    :param layer: Default None. If specified, load only the faults for the layer specified
    :return: dataframe containing all the bit-flip faults and their effect
    """

    image_merge_folder = f'../fault_injection/{net_name}/'

    df_list = []
    with tqdm(total=number_of_chunks) as pbar:
        for filename in os.listdir(image_merge_folder):

            golden_filename = f'../golden/{net_name}/{filename.replace("exhaustive_results", "golden")}'
            golden_df = pd.read_csv(golden_filename)

            for chunk in pd.read_csv(f'{image_merge_folder}/{filename}', chunksize=chunksize):
                pbar.set_description(f'Loading {filename}')

                if layer is not None:
                    chunk = chunk[chunk.Layer == layer]

                chunk = chunk.dropna()
                if 'Golden' in chunk.columns:
                    chunk = chunk.drop('Golden', axis=1).merge(golden_df[['ImageIndex', 'Golden']], how='left',
                                                               on='ImageIndex')
                chunk = chunk[['Bit', 'Layer', 'Top_1', 'Golden']]
                df_list.append(chunk)

                stuck_at_chunk = copy.deepcopy(chunk)
                stuck_at_chunk.Top_1 = stuck_at_chunk.Golden
                df_list.append(stuck_at_chunk)

                pbar.update(1)

    image_df = pd.concat(df_list, ignore_index=True)

    return image_df


def _load_intermediate_results(load_folder, layer, seed, number_biased_samples, number_unbiased_samples, number_date_per_layer_samples, number_date_samples):

    biased_sample_p_n = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_biased_samples}_biased_sample_p_n.npy')
    biased_sample_error = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_biased_samples}_biased_sample_error.npy')

    unbiased_sample_p_n = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_unbiased_samples}_unbiased_sample_p_n.npy')
    unbiased_sample_error = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_unbiased_samples}_unbiased_sample_error.npy')

    date_per_layer_p_n = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_per_layer_samples}_date_per_layer_p_n.npy')
    date_per_layer_error = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_per_layer_samples}_date_per_layer_error.npy')

    date_p_n = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_samples}_date_p_n.npy')
    date_error = np.load(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_samples}_date_error.npy')

    exhaustive_p = np.load(f'{load_folder}/{str(layer).zfill(2)}/exhaustive_p.npy')

    intermediate_results = {
        'biased_sample_p_n': biased_sample_p_n,
        'biased_sample_error': biased_sample_error,
        'unbiased_sample_p_n': unbiased_sample_p_n,
        'unbiased_sample_error': unbiased_sample_error,
        'date_per_layer_p_n': date_per_layer_p_n,
        'date_per_layer_error': date_per_layer_error,
        'date_p_n': date_p_n,
        'date_error': date_error,
        'exhaustive_p': exhaustive_p
    }

    return intermediate_results


def return_critical_percentage_by_bit(image_df):
    total_critical = len(image_df[image_df.Top_1 != image_df.Golden])

    result_dicts = []
    for bit in np.arange(0, 32):
        complete_df_bit = copy.deepcopy(image_df[image_df.Bit == bit])

        if total_critical == 0:
            critical = 0
        else:
            critical = np.sum(complete_df_bit.Top_1 != complete_df_bit.Golden) / total_critical
        masked = 1 - critical

        result_dicts.append({'Bit': bit,
                             'Masked': masked * 100,
                             'Critical': critical * 100})
    return result_dicts


def return_complete_df(image_df, avoid_mantissa=False, mantissa_injections=None):

    result_dicts = []

    if avoid_mantissa:
        result_dicts.append({'Bit': 0,
                             'Masked': 100,
                             'Critical': 0,
                             'InjectedFaults': mantissa_injections})

    for bit in np.arange(0, 32):
        complete_df_bit = copy.deepcopy(image_df[image_df.Bit == bit])

        # We are actually doing bit-flips since all the stuck-at that do not change value are always masked
        injected_faults = len(complete_df_bit)
        if injected_faults == 0:
            masked = 1
        else:
            masked = len(complete_df_bit[complete_df_bit.Top_1 == complete_df_bit.Golden]) / injected_faults
        critical = 1 - masked

        result_dicts.append({'Bit': bit,
                             'Masked': masked * 100,
                             'Critical': critical * 100,
                             'InjectedFaults': injected_faults})
    return result_dicts


def create_sample(df, n, random_state=None, avoid_mantissa=False):
    if isinstance(n, int):
        if avoid_mantissa:
            n = int(n - (23 * n/32))
        return df.sample(n, axis=0, random_state=random_state)

    sample_df_list = []

    bit_start = 23 if avoid_mantissa else 0

    for bit in np.arange(bit_start, 32):
        sample_df_list.append(df[df.Bit == bit].sample(n=int(n[bit]), axis=0, random_state=random_state))

    return pd.concat(sample_df_list)


def return_p_and_n(df, convert_p_to_probability=True):
    p = df.Critical.mean() / 100 if convert_p_to_probability else df.Critical.mean()
    n = df.InjectedFaults.sum()

    return p, n


def extract_sample_p_n_bit(net_name,
                           layer_number,
                           number_biased_samples,
                           number_unbiased_samples,
                           number_date_per_layer_samples,
                           number_date_samples,
                           number_of_samples_folder='../fault_injection/number_of_samples',
                           load_if_exist=True,
                           load_folder='intermediate_results/bit',
                           seed=1234,
                           avoid_mantissa=False):

    load_folder = f'{load_folder}/{net_name}'
    os.makedirs(f'{load_folder}/{str(layer_number).zfill(2)}', exist_ok=True)

    try:
        if not load_if_exist:
            raise FileNotFoundError

        intermediate_results = _load_intermediate_results(load_folder,
                                                          layer_number,
                                                          seed,
                                                          number_biased_samples,
                                                          number_unbiased_samples,
                                                          number_date_per_layer_samples,
                                                          number_date_samples)
    except FileNotFoundError:

        intermediate_results = {}
        random_state = np.random.default_rng(seed=seed).bit_generator

        # load N from file
        number_of_samples_folder = f'{number_of_samples_folder}/{net_name}'
        N = pd.read_csv(f'{number_of_samples_folder}/N_layers.csv', index_col=0)['N']

        # Load dataframe of all faults
        number_of_chunks = 1000
        chunksize = int((100 * N.values.sum() / 2) / number_of_chunks)
        image_df = _create_df(net_name,
                              number_of_chunks=number_of_chunks,
                              chunksize=chunksize,
                              layer=layer_number)

        # Data-aware
        biased_n_by_bit = pd.read_csv(f'{number_of_samples_folder}/n_tuning_scenario_1.csv', index_col=0).loc[layer_number].to_list()
        biased_n_by_bit.reverse()

        biased_sample_list = [
            pd.DataFrame(return_critical_percentage_by_bit(create_sample(image_df,
                                                                         n=biased_n_by_bit,
                                                                         random_state=random_state,
                                                                         avoid_mantissa=avoid_mantissa),)).set_index('Bit')
            for _ in tqdm(np.arange(0, number_biased_samples), desc='Sampling biased')]
        intermediate_results['biased_sample_p'] = biased_sample_list

        # Date per layer
        date_per_layer_n = int(
            pd.read_csv(f'{number_of_samples_folder}/n_date_per_layer.csv', index_col=0, header=None).loc[layer_number].values[0])

        date_per_layer_list = [
            pd.DataFrame(return_critical_percentage_by_bit(create_sample(image_df,
                                                                         n=date_per_layer_n,
                                                                         random_state=random_state,
                                                                         avoid_mantissa=avoid_mantissa))).set_index('Bit')
            for _ in tqdm(np.arange(0, number_date_per_layer_samples), desc='Sampling DATE09 per layer')]
        intermediate_results['date_per_layer_p'] = date_per_layer_list

        # Exhaustive
        results_dataframe_complete = pd.DataFrame(return_critical_percentage_by_bit(image_df))
        results_dataframe_complete = results_dataframe_complete.set_index('Bit')
        intermediate_results['exhaustive_p'] = results_dataframe_complete

    return intermediate_results


def extract_p_per_bit(net_name,
                      number_of_samples_folder='../fault_injection/number_of_samples'):

    number_of_samples_folder = f'{number_of_samples_folder}/{net_name}'
    N = pd.read_csv(f'{number_of_samples_folder}/N_layers.csv', index_col=0)['N']

    number_of_chunks = 1000
    chunksize = int((100 * N.values.sum() / 2) / number_of_chunks)

    image_df = _create_df(net_name,
                          number_of_chunks=number_of_chunks,
                          chunksize=chunksize)

    p_per_bit_df = pd.DataFrame(return_complete_df(image_df))
    p_per_bit_df = p_per_bit_df.set_index('Bit')

    return p_per_bit_df


def extract_samples_p_n(layer_start,
                        layer_end,
                        net_name,
                        number_biased_samples,
                        number_unbiased_samples,
                        number_date_per_layer_samples,
                        number_date_samples,
                        number_of_samples_folder='../fault_injection/number_of_samples',
                        load_if_exist=True,
                        load_folder='intermediate_results',
                        seed=1234,
                        avoid_mantissa=False):

    load_folder = f'{load_folder}/{net_name}'
    os.makedirs(load_folder, exist_ok=True)

    # load N from file
    number_of_samples_folder = f'{number_of_samples_folder}/{net_name}'
    N = pd.read_csv(f'{number_of_samples_folder}/N_layers.csv', index_col=0)['N']

    biased_sample_p_n = {}
    biased_sample_error = {}

    unbiased_sample_p_n = {}
    unbiased_sample_error = {}

    date_per_layer_p_n = {}
    date_per_layer_error = {}

    date_p_n = {}
    date_error = {}

    exhaustive_p = {}

    try:
        if not load_if_exist:
            raise FileNotFoundError

        for layer in np.arange(layer_start, layer_end):
            intermediate_results = _load_intermediate_results(load_folder,
                                                              layer,
                                                              seed,
                                                              number_biased_samples,
                                                              number_unbiased_samples,
                                                              number_date_per_layer_samples,
                                                              number_date_samples)

            biased_sample_p_n[layer] = (intermediate_results['biased_sample_p_n'])
            biased_sample_error[layer] = (intermediate_results['biased_sample_error'])

            unbiased_sample_p_n[layer] = (intermediate_results['unbiased_sample_p_n'])
            unbiased_sample_error[layer] = (intermediate_results['unbiased_sample_error'])

            date_per_layer_p_n[layer] = (intermediate_results['date_per_layer_p_n'])
            date_per_layer_error[layer] = (intermediate_results['date_per_layer_error'])

            date_p_n[layer] = (intermediate_results['date_p_n'])
            date_error[layer] = (intermediate_results['date_error'])

            exhaustive_p[layer] = (intermediate_results['exhaustive_p'])

        print('Loaded from saved files...')

    except FileNotFoundError as not_found:

        print(f'File {not_found.filename} not found, computing results...')

        number_of_chunks = 1000
        chunksize = int((100 * N.values.sum() / 2) / number_of_chunks)

        image_df = _create_df(net_name,
                              number_of_chunks=number_of_chunks,
                              chunksize=chunksize)

        random_state = np.random.default_rng(seed=seed).bit_generator

        for layer in np.arange(layer_start, layer_end):
            print(f'#### Layer {layer} ####')
            os.makedirs(f'{load_folder}/{str(layer).zfill(2)}', exist_ok=True)

            image_df_layer = image_df[image_df.Layer.astype(int) == layer]
            N_layer = N[layer]

            # Biased with critical
            biased_n_by_bit = pd.read_csv(f'{number_of_samples_folder}/n_tuning_scenario_1.csv', index_col=0).loc[layer].to_list()
            biased_n_by_bit.reverse()

            biased_sample_list = [
                pd.DataFrame(return_complete_df(create_sample(image_df_layer,
                                                              n=biased_n_by_bit,
                                                              random_state=random_state,
                                                              avoid_mantissa=avoid_mantissa),
                                                avoid_mantissa=avoid_mantissa,
                                                mantissa_injections=np.sum(biased_n_by_bit[0:23]))).set_index('Bit')
                for _ in tqdm(np.arange(0, number_biased_samples), desc='Sampling biased')]
            biased_sample_p_n[layer] = ([return_p_and_n(biased_sample) for biased_sample in biased_sample_list])
            biased_sample_error[layer] = ([compute_error_margin(p, n, N_layer, finite_population=True) for p, n in biased_sample_p_n[layer]])
            np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_biased_samples}_biased_sample_p_n.npy',  np.asarray(biased_sample_p_n[layer]))
            np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_biased_samples}_biased_sample_error.npy',  np.asarray(biased_sample_error[layer]))

            # Unbiased
            unbiased_n_by_bit = pd.read_csv(f'{number_of_samples_folder}/n_worst_case.csv', index_col=0).loc[layer].to_list()
            unbiased_n_by_bit.reverse()

            unbiased_sample_list = [
                pd.DataFrame(return_complete_df(create_sample(image_df_layer,
                                                              n=unbiased_n_by_bit,
                                                              random_state=random_state,
                                                              avoid_mantissa=avoid_mantissa),
                                                avoid_mantissa=avoid_mantissa,
                                                mantissa_injections=np.sum(unbiased_n_by_bit[0:23]))).set_index('Bit')
                for _ in tqdm(np.arange(0, number_unbiased_samples), desc='Sampling unbiased')]
            unbiased_sample_p_n[layer] = ([return_p_and_n(unbiased_sample) for unbiased_sample in unbiased_sample_list])
            unbiased_sample_error[layer] = ([compute_error_margin(p, n, N_layer) for p, n in unbiased_sample_p_n[layer]])
            np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_unbiased_samples}_unbiased_sample_p_n.npy',  np.asarray(unbiased_sample_p_n[layer]))
            np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_unbiased_samples}_unbiased_sample_error.npy',  np.asarray(unbiased_sample_error[layer]))

            # Date per layer
            date_per_layer_n = int(pd.read_csv(f'{number_of_samples_folder}/n_date_per_layer.csv', index_col=0, header=None).loc[layer].values[0])

            date_per_layer_list = []
            for _ in tqdm(np.arange(0, number_date_per_layer_samples), desc='Sampling DATE09 per layer'):
                sample = create_sample(image_df_layer,
                                       n=date_per_layer_n,
                                       random_state=random_state,
                                       avoid_mantissa=avoid_mantissa)
                date_per_layer_list.append(pd.DataFrame(return_complete_df(sample,
                                                                           avoid_mantissa=avoid_mantissa,
                                                                           mantissa_injections=date_per_layer_n - len(sample))).set_index('Bit'))
            date_per_layer_p_n[layer] = [return_p_and_n(date_per_layer) for date_per_layer in date_per_layer_list]
            date_per_layer_error[layer] = [compute_error_margin(p, n, N_layer) for p, n in date_per_layer_p_n[layer]]
            np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_per_layer_samples}_date_per_layer_p_n.npy', np.asarray(date_per_layer_p_n[layer]))
            np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_per_layer_samples}_date_per_layer_error.npy', np.asarray(date_per_layer_error[layer]))

            # Date
            date_n = int(pd.read_csv(f'{number_of_samples_folder}/n_date.csv', index_col=0, header=None).loc[layer].values[0])

            date_list = []
            for _ in tqdm(np.arange(0, number_date_samples), desc='Sampling DATE09'):
                sample = create_sample(image_df_layer,
                                       n=date_n,
                                       random_state=random_state,
                                       avoid_mantissa=avoid_mantissa)
                date_list.append(pd.DataFrame(return_complete_df(sample,
                                                                 avoid_mantissa=avoid_mantissa,
                                                                 mantissa_injections=date_n - len(sample))).set_index('Bit'))
            date_p_n[layer] = [return_p_and_n(date) for date in date_list]
            date_error[layer] = [compute_error_margin(p, n, N_layer) for p, n in date_p_n[layer]]
            np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_samples}_date_p_n.npy', np.asarray(date_p_n[layer]))
            np.save(f'{load_folder}/{str(layer).zfill(2)}/{seed}_{number_date_samples}_date_error.npy', np.asarray(date_error[layer]))

            # Exhaustive results
            print('Extracting exhaustive p...')
            results_dataframe_complete = pd.DataFrame(return_complete_df(image_df_layer))
            results_dataframe_complete = results_dataframe_complete.set_index('Bit')
            exhaustive_p[layer] = results_dataframe_complete.Critical.values
            np.save(f'{load_folder}/{str(layer).zfill(2)}/exhaustive_p.npy',  np.asarray(exhaustive_p[layer]))

    return [biased_sample_p_n, biased_sample_error], [unbiased_sample_p_n, unbiased_sample_error], [date_per_layer_p_n, date_per_layer_error], [date_p_n, date_error], exhaustive_p
