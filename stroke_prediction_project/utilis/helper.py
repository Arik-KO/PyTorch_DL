import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import seaborn as sns

def plot_visualization_eda(something:NDArray, another_thing: NDArray, label_1:str, label_2:str, figure_name:str, save_plot:bool ):
    plt.figure(figsize = (8,5), dpi = 150)
    plt.plot(something, linewidth = 2, color = 'royalblue', label = label_1)
    plt.plot(another_thing, linewidth = 2, linestyle = '--', color = 'coral', label = label_2)
    plt.grid()
    plt.legend(loc = 1)
    plt.tight_layout()
    if save_plot:
        plt.savefig(f'../results/plots/{figure_name}.png', dpi = 200)
    plt.show()


def losses_plot(train_loss:NDArray, val_loss: NDArray, fig_name:str, save_plot:str, label_1 = 'train_loss', label_2 = 'val_loss' ):
    plt.figure(figsize = (8,5), dpi = 150)
    plt.plot(train_loss, linewidth = 2, color = 'coral', label = label_1)
    plt.plot(val_loss, linewidth = 2, linestyle = '--', color = 'royalblue', label = label_2)
    plt.grid()
    plt.legend(loc = 1)
    plt.tight_layout()
    if save_plot:
        plt.savefig(f'../results/plots/{fig_name}.png', dpi = 200)
    plt.show()


def corr_visualization(data_matrix:pd.DataFrame, annotation:bool,  fig_name:str, save_heatmap:bool ):
    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(data_matrix, annot=annotation, cmap='coolwarm', vmin=-1, vmax=1)
    plt.tight_layout()
    if save_heatmap:
        plt.savefig(f'../results/plots/{fig_name}.png', dpi = 200)
    plt.show()