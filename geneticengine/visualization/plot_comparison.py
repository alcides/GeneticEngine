import glob
import pandas as pd
from pylab import *
from palettable.colorbrewer.qualitative import Set2_7

def load(example_name, metric, single_value):
    f_list = glob.glob(f'results\csvs\{example_name}_(seed=*).csv')

    data = list()

    for f in f_list:
        df = pd.read_csv(f)
        if not single_value:
            df = df[[metric, 'number_of_the_generation']]
            df = df.groupby('number_of_the_generation').agg('mean')
        data.append(df[metric].values)
    return np.array(data)

def med(data):
    median = np.zeros(data.shape[1])
    for i in range(0, len(median)):
        median[i] = np.median(data[:, i])
    return median

def perc(data):
   median = np.zeros(data.shape[1])
   perc_25 = np.zeros(data.shape[1])
   perc_75 = np.zeros(data.shape[1])
   for i in range(0, len(median)):
       median[i] = np.median(data[:, i])
       perc_25[i] = np.percentile(data[:, i], 25)
       perc_75[i] = np.percentile(data[:, i], 75)
   return median, perc_25, perc_75
    

def plot_comparison(file_name1, file_name2, name1, name2, result_name='results/images/medians.png', metric='fitness', single_value=False):
    example1_data = load(file_name1,metric=metric,single_value=single_value)
    example2_data = load(file_name2,metric=metric,single_value=single_value)

    n_generations = example1_data.shape[1]
    x = np.arange(0, n_generations)

    med_example1, perc_25_example1, perc_75_example1 = perc(example1_data)
    med_example2, perc_25_example2, perc_75_example2 = perc(example2_data)


    colors = Set2_7.mpl_colors
    axes(frameon=0)
    grid(axis='y', color="0.9", linestyle='-', linewidth=1)

    plot(x, med_example1, linewidth=2, color=colors[0])
    plot(x, med_example2, linewidth=2, linestyle='--', color=colors[1])

    fill_between(x, perc_25_example1, perc_75_example1, alpha=0.25, linewidth=0, color=colors[0])
    fill_between(x, perc_25_example2, perc_75_example2, alpha=0.25, linewidth=0, color=colors[1])

    legend = plt.legend([name1, name2], loc=4)
    frame = legend.get_frame()
    frame.set_facecolor('1.0')
    frame.set_edgecolor('1.0')

    savefig(result_name)

plot_comparison('all\GoL', 'all\GoL_vectorial', 'normal GoL', 'vectorial GoL')

