# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class CalcEnergy:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def calc_energy(self, data):
        data = np.array(data)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        m = data.shape[0]
        energy_n = np.zeros((m, 1))
        for ii in range(m - self.alpha - self.beta):
            energy_n[ii + self.alpha + self.beta] = 1 / self.alpha * np.sum(
                np.sum(np.abs(data[ii + self.beta:ii + self.beta + self.alpha, :] - data[ii:ii + self.alpha, :]), 1), 0)
            diff_mat = data[ii + self.beta:ii + self.beta + self.alpha, :] - data[ii:ii + self.alpha, :]
            diff_mat_max_sub = np.array(np.where(np.abs(diff_mat) == np.max(np.abs(diff_mat))))
            diff_mat_max_sub = diff_mat_max_sub[:, 0]
            max_sign = np.sign(diff_mat[diff_mat_max_sub[0], diff_mat_max_sub[1]])
            energy_n[ii + self.alpha + self.beta] *= max_sign
        return energy_n.T[0]


if __name__ == '__main__':
    data = pd.read_excel("①-胶头密集打36点(l2r)-1100g.xlsx")
    data = np.array(data)
    np.savetxt("test2.txt", data)
    ce = CalcEnergy(3, 5)
    ret = ce.calc_energy(data)
    plt.plot(data)
    plt.plot(ret)
    plt.show()
