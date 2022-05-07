from matplotlib import pyplot as plt
import numpy as np
import os


class Plotter:

    def __init__(self):
        self.exhaustive_color = '#b33939'
        self.color_bar_critical_1 = '#2c2c54'
        self.color_bar_critical_2 = '#ff793f'
        # color_bar_n_1 = '#706fd3'
        self.color_bar_n_1 = '#33d9b2'
        self.color_bar_n_2 = '#33d9b2'
        self.color_bar_n_2_1 = '#33d9b2'

    def _plot(self,
              sample_p,
              exhaustive_p,
              sample_name):

        fig, ax1 = plt.subplots(1, figsize=(12, 6))
        width = .5

        x_1 = np.arange(0, len(sample_p))

        p_mean = [np.mean(layer) for layer in sample_p]
        p_max = [np.max(layer) for layer in sample_p]
        p_min = [np.min(layer) for layer in sample_p]
        y_error = np.array([np.array(p_mean) - np.array(p_min), np.array(p_max) - np.array(p_mean)])

        color = self.color_bar_critical_1 if sample_name == 'data_aware' else self.color_bar_n_1 if sample_name == 'data_unaware' else self.color_bar_critical_2
        ax1.bar(x=x_1,
                height=p_mean,
                yerr=y_error,
                error_kw=dict(ecolor='grey', alpha=.95, lw=1.25, capsize=2, capthick=.5, zorder=2),
                width=width,
                color=self.color_bar_critical_1,
                label='Proposed (left)',
                zorder=1)

        ax1.plot(x_1,
                 exhaustive_p,
                 linestyle='--',
                 color=self.exhaustive_color)

        ax1.set_ylim(0, 1.6)

        fig.show()
        os.makedirs('plot/DATEvsProposed', exist_ok=True)
        fig.savefig(f'plot/DATEvsProposed/layer_analysis_{sample_name}.png')

    def plot_layers(self,
                    biased_sample_p_n,
                    unbiased_sample_p_n,
                    date_per_layer_p_n,
                    date_p_n,
                    exhaustive_p):

        self._plot(sample_p=[[p * 100 for p, n in layer] for layer in biased_sample_p_n],
                   exhaustive_p=exhaustive_p,
                   sample_name='data_aware')
        self._plot(sample_p=[[p * 100 for p, n in layer] for layer in unbiased_sample_p_n],
                   exhaustive_p=exhaustive_p,
                   sample_name='data_unaware')
        self._plot(sample_p=[[p * 100 for p, n in layer] for layer in date_per_layer_p_n],
                   exhaustive_p=exhaustive_p,
                   sample_name='date_per_layer')
        self._plot(sample_p=[[p * 100 for p, n in layer] for layer in date_p_n],
                   exhaustive_p=exhaustive_p,
                   sample_name='date')

    def plot_graph(self,
                   biased_sample_p_n,
                   biased_sample_error,
                   date_per_layer_p_n,
                   date_per_layer_error,
                   exhaustive_p,
                   layer,
                   net_name,
                   mode='complete',
                   number_biased_samples=10,
                   number_per_layer_samples=10,
                   critical_top=10):

        total_number_of_samples = number_biased_samples + number_per_layer_samples

        fig, ax1 = plt.subplots(1, figsize=(12, 6))

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
                    color=self.color_bar_critical_1,
                    label='Proposed (left)',
                    zorder=1)
            ax1.bar(x=x_2,
                    height=[p * 100 for p, n in date_per_layer_p_n],
                    yerr=[error * 100 for error in date_per_layer_error],
                    error_kw=dict(ecolor='black', alpha=.75, lw=1.25, capsize=2, capthick=.5, zorder=2),
                    width=width,
                    color=self.color_bar_critical_2,
                    label='DATE09 per layer (left)',
                    zorder=1)

            ax1.axhline(exhaustive_p, linestyle='--', linewidth=1, color=self.exhaustive_color, zorder=99999,
                        label='Exhaustive FI results (left)')

            yticks = [i for i in np.linspace(0, critical_top, 6)]
            yticks.append(exhaustive_p)
            yticks_labels = [f'{i:.2f}' for i in yticks]
            ax1.set_ylim(0, critical_top)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticks_labels)
            ax1.set_ylabel('Critical Faults [%]')
            ax1.get_yticklabels()[-1].set_color(self.exhaustive_color)

            if mode == 'complete':
                ax1.bar(x=0,
                        height=0,
                        width=0,
                        color=self.color_bar_n_1,
                        label='Injected Faults (right)')

            ax1.legend(loc='upper left')

        if mode == 'complete' or mode == 'injected':
            ax2.bar(x=x_3,
                    height=[n / 10000 for p, n in biased_sample_p_n],
                    width=width,
                    color=self.color_bar_n_1,
                    label='Proposed')
            ax2.bar(x=x_4,
                    height=[n / 10000 for p, n in date_per_layer_p_n],
                    width=width,
                    color=self.color_bar_n_2_1,
                    label='DATE09 per layer')

            yticks = [i for i in np.linspace(0, 1.5, 7)]
            yticks_labels = [f'{int(i * 10000): ,}' for i in yticks]
            ax2.set_yticks(yticks)
            ax2.set_yticklabels(yticks_labels)
            ax2.set_ylabel('Injected Faults (n)')

        if mode == 'injected':
            ax2.legend(loc='upper left')

        fig.show()
        os.makedirs(f'plot/DATEvsProposed/{net_name}', exist_ok=True)
        fig.savefig(f'plot/DATEvsProposed/{net_name}/{str(layer).zfill(2)}_{mode}_critical.png')