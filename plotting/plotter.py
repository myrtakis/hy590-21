import matplotlib.pyplot as plt
import dc_stat_think as dcst


def plot_observed_null_distr(sttc_matrix, sig_threshold=4):
    sttc_matrix = sttc_matrix[sttc_matrix['zscore'] > sig_threshold]
    plt.hist(sttc_matrix['STTC'], bins='auto', alpha=0.5, label='Observed STTC')
    plt.hist(sttc_matrix['NullSTTC'], bins='auto', alpha=0.5, label='Null STTC')
    plt.legend()
    plt.xlabel('STTC Value')
    plt.show()


def plot_ecdf(data, xlabel):
    x, y = dcst.ecdf(data)
    plt.plot(x, y*100)
    plt.xlabel(xlabel, size=14)
    plt.ylabel('Percent', size=14)
    plt.show()


def plot_zscore(zscore_data):
    significant_edges_perc = []
    for t in range(11):
        significant_edges_perc.append((sum(zscore_data > t) / len(zscore_data)) * 100)
    plt.plot(significant_edges_perc)
    plt.ylabel('Percentage of edges above threshold (%)', fontsize=14)
    plt.xlabel('Z score threshold', fontsize=14)
    plt.show()
