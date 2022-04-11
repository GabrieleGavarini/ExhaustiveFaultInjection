from utils import plot_graph, extract_samples_p_n

# color_bar_critical_1 = '#5b8ba8'
# color_bar_critical_2 = '#1c506e'
# color_bar_n_1 = '#f3880a'
# color_bar_n_2 = '#f3880a'

layer = 1
number_biased_samples = 10
number_per_layer_samples = 10

biased_sample, unbiased_sample, date_per_layer, exhaustive_p = extract_samples_p_n(layer=layer,
                                                                                   net_name='resnet20',
                                                                                   number_biased_samples=number_biased_samples,
                                                                                   number_unbiased_samples=0,
                                                                                   number_per_layer_samples=number_per_layer_samples,
                                                                                   seed=51195,
                                                                                   load_if_exist=False)

plot_graph(biased_sample_p_n=biased_sample[0],
           biased_sample_error=biased_sample[1],
           date_per_layer_p_n=date_per_layer[0],
           date_per_layer_error=date_per_layer[1],
           exhaustive_p=exhaustive_p,
           layer=layer,
           mode='faults',
           number_biased_samples=number_biased_samples,
           number_per_layer_samples=number_per_layer_samples)
plot_graph(biased_sample_p_n=biased_sample[0],
           biased_sample_error=biased_sample[1],
           date_per_layer_p_n=date_per_layer[0],
           date_per_layer_error=date_per_layer[1],
           exhaustive_p=exhaustive_p,
           layer=layer,
           mode='injected',
           number_biased_samples=number_biased_samples,
           number_per_layer_samples=number_per_layer_samples)
plot_graph(biased_sample_p_n=biased_sample[0],
           biased_sample_error=biased_sample[1],
           date_per_layer_p_n=date_per_layer[0],
           date_per_layer_error=date_per_layer[1],
           exhaustive_p=exhaustive_p,
           layer=layer,
           mode='complete',
           number_biased_samples=number_biased_samples,
           number_per_layer_samples=number_per_layer_samples)
