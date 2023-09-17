import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


if __name__ == '__main__':
    plt.figure(figsize=(8, 6.5))
    window_size = 5
    model_learning_period = 1000
    evaluation_cycle = 50
    scale = model_learning_period / evaluation_cycle

    # If the data is obtained without the model-aided method, because we plot in log scale,
    # to make the plot smooth, we need to add some data points. These data points are obtained during evaluation.
    dir_1_1 = 'dir_1_1'  # the directory of the data obtained with algorithm IQL
    data_1_1 = np.load(dir_1_1 + '/episode_data.npy') / 200000
    data_1_1 = np.concatenate([data_1_1[:100], data_1_1[100:1000:50], data_1_1[1000:1580],])
    # For the data obtained with random runs, you can also import the data with above code similarly,
    # e.g., data_1_2, data_1_3
    dir_1_2 = 'dir_1_2'
    data_1_2 = np.load(dir_1_2 + '/episode_data.npy') / 200000
    data_1_2 = np.concatenate([data_1_2[:100], data_1_2[100:1000:50], data_1_2[1000:1580],])
    dir_1_3 = 'dir_1_3'
    data_1_3 = np.load(dir_1_3 + '/episode_data.npy') / 200000
    data_1_3 = np.concatenate([data_1_3[:100], data_1_3[100:1000:50], data_1_3[1000:1580],])

    dir_2_1 = 'dir_2_1'  # the directory of the data obtained with algorithm QMIX
    data_2_1 = np.load(dir_2_1 + '/episode_data.npy') / 200000
    data_2_1 = np.concatenate([data_2_1[:100], data_2_1[100:1000:50], data_2_1[1000:1580],])
    # Use the same code to import the data obtained with other random runs
    # Here we just use None to represent the processed data
    data_2_2 = None
    data_2_3 = None

    # The data is obtained with the model-aided method
    dir_3_1 = 'dir_3_1'  # the directory of the data obtained with algorithm model-aided QMIX
    data_3_1 = np.load(dir_3_1 + '/episode_data.npy')[:600] / 200000
    # Use the same code to import the data obtained with other random runs
    data_3_2 = None
    data_3_3 = None

    dir_4_1 = 'dir_4_1'  # the directory of the data obtained with algorithm model-aided FedQMIX
    data_4_1 = np.load(dir_4_1 + '/global_data.npy')[:600] / 200000
    # Use the same code to import the data obtained with other random runs
    data_4_2 = None
    data_4_3 = None

    # Calculate the mean and confidence interval of the data obtained without the model-aided method
    mean_data_1 = np.mean([data_1_1, data_1_2, data_1_3], axis=0)
    low_quantile_data_1 = np.percentile([data_1_1, data_1_2, data_1_3], 0.5, axis=0)
    high_quantile_data_1 = np.percentile([data_1_1, data_1_2, data_1_3], 99.5, axis=0)
    mv_mean_data_1 = moving_average(mean_data_1, 5)
    mv_low_quantile_data_1 = moving_average(low_quantile_data_1, window_size)
    mv_high_quantile_data_1 = moving_average(high_quantile_data_1, window_size)
    mv_x_axis_1 = list(range(len(mv_mean_data_1)))
    mv_x_axis_1 = mv_x_axis_1[0:101] + [(x-98) * 50 for x in mv_x_axis_1[101:]]

    # Use the same code as above the calculate the mean and confidence interval of data obtained with QMIX
    mean_data_2 = np.mean([data_2_1, data_2_2, data_2_3], axis=0)
    low_quantile_data_2 = np.percentile([data_2_1, data_2_2, data_2_3], 0.5, axis=0)
    high_quantile_data_2 = np.percentile([data_2_1, data_2_2, data_2_3], 99.5, axis=0)
    mv_mean_data_2 = moving_average(mean_data_2, 5)
    mv_low_quantile_data_2 = moving_average(low_quantile_data_2, window_size)
    mv_high_quantile_data_2 = moving_average(high_quantile_data_2, window_size)
    mv_x_axis_2 = list(range(len(mv_mean_data_2)))
    mv_x_axis_2 = mv_x_axis_1[0:101] + [(x-98) * 50 for x in mv_x_axis_1[101:]]

    # Calculate the mean and confidence interval of the data obtained with the model-aided method
    mean_data_3 = np.mean([data_3_1, data_3_2, data_3_3], axis=0)
    low_quantile_data_3 = np.percentile([data_3_1, data_3_2, data_3_3], 0.5, axis=0)
    high_quantile_data_3 = np.percentile([data_3_1, data_3_2, data_3_3], 99.5, axis=0)
    mv_mean_data_3 = moving_average(mean_data_3, 5)
    mv_low_quantile_data_3 = moving_average(low_quantile_data_3, window_size)
    mv_high_quantile_data_3 = moving_average(high_quantile_data_3, window_size)
    mv_x_axis_3 = list(range(len(mv_mean_data_3)))
    mv_x_axis_3 = [x / scale for x in mv_x_axis_3]  # scale the x-axis because we use the model

    # Use the same code as above the calculate the mean and confidence interval of model-aided FedQMIX
    mean_data_4 = np.mean([data_4_1, data_4_2, data_4_3], axis=0)
    low_quantile_data_4 = np.percentile([data_4_1, data_4_2, data_4_3], 0.5, axis=0)
    high_quantile_data_4 = np.percentile([data_4_1, data_4_2, data_4_3], 99.5, axis=0)
    mv_mean_data_4 = moving_average(mean_data_4, 5)
    mv_low_quantile_data_4 = moving_average(low_quantile_data_4, window_size)
    mv_high_quantile_data_4 = moving_average(high_quantile_data_4, window_size)
    mv_x_axis_4 = list(range(len(mv_mean_data_4)))
    mv_x_axis_4 = [x / scale for x in mv_x_axis_4]

    #  Plot the results with confidence band
    plt.plot(mv_x_axis_4, mv_mean_data_4, color='r', linewidth=4, label='Model-aided FedQMIX')
    plt.fill_between(mv_x_axis_4, mv_low_quantile_data_4, mv_high_quantile_data_4, color='r', alpha=0.1)

    plt.plot(mv_x_axis_3, mv_mean_data_3, color='#0082FF', linewidth=4, label='Model-aided QMIX', linestyle='--')
    plt.fill_between(mv_x_axis_3, mv_low_quantile_data_3, mv_high_quantile_data_3, color='#0082FF', alpha=0.1)

    plt.plot(mv_x_axis_2, mv_mean_data_2, color='g', linewidth=4, label='QMIX (model-free)', linestyle='-.')
    plt.fill_between(mv_x_axis_2, mv_low_quantile_data_2, mv_high_quantile_data_2, color='g', alpha=0.1)

    plt.plot(mv_x_axis_1, mv_mean_data_1, color='#FF8200', linewidth=4, label='IQL (model-free)', linestyle=':')
    plt.fill_between(mv_x_axis_1, mv_low_quantile_data_1, mv_high_quantile_data_1, color='#FF8200', alpha=0.1)

    plt.xscale("log")
    plt.grid()
    plt.xlim([0.11, 30000])
    plt.ylim([0.2, 1.0])
    # font size for GlobeCom, linewidth=4
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Episode [log scale]', fontsize=22)
    plt.ylabel('Data collection ratio', fontsize=22)
    plt.legend(loc='lower right', fontsize=18)
    # font size for pre, linewidth=3
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.xlabel('Episode [log scale]', fontsize=15)
    # plt.ylabel('Data collection ratio', fontsize=15)
    # plt.legend(loc='lower right', fontsize=15)
    result_dir = '/Figures'
    plt.savefig(result_dir + '/results.pdf', format='pdf', bbox_inches = 'tight',pad_inches = 0.1)
    plt.savefig(result_dir + '/results.png', bbox_inches = 'tight',pad_inches = 0.1)
    plt.show()