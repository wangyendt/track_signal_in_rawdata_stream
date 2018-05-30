# encoding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import sys


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


def preprocess(data):
    if np.ndim(data) == 1:
        data = data - data[0]
    else:
        data = data - data[1, :]
    return data


def calc_moving_avg(data, winlen):
    output = []
    for ii in range(len(data)):
        if ii <= winlen:
            output.append(np.mean(data[:ii + 1]))
        else:
            output.append(np.mean(data[ii - (winlen - 1):ii + 1]))
    return np.array(output)


def calc_energy(data, alpha, beta):
    ce = CalcEnergy(alpha, beta)
    energy = ce.calc_energy(data)
    return energy


def calc_flag(energy, thd, step):
    flag = np.zeros(energy.shape)
    for ii in range(flag.shape[0]):
        if ii > 0:
            if flag[ii - 1] == 0:
                if energy[ii] > thd[0]:
                    flag[ii] = 1
                    thd[0] = thd[0] + step[0]
                    continue
                else:
                    flag[ii] = 0
            if flag[ii - 1] == 1:
                if energy[ii] < thd[1]:
                    flag[ii] = 0
                    thd[1] = thd[1] + step[1]
                    continue
                else:
                    flag[ii] = 1
    return flag


def show_fig(data, energy, flag, params):
    fig = plt.figure()
    fig.set_size_inches(60, 10)
    plt.plot(data, '-', linewidth=3)
    plt.plot(energy)
    plt.hlines(params[0], 0, data.shape[0], linestyles='--')
    plt.hlines(params[1], 0, data.shape[0], linestyles='--')
    plt.plot(flag * (np.max(data) - np.min(data)) + np.min(data), '--')
    plt.legend(['rawdata1', 'rawdata2', 'energy', 'upper limit', 'lower limit', 'touch flag'])
    plt.show()


def calc_force(data_, flag_, bfm_len, aft_len, avg_len, force):
    tds = np.array(np.where(np.diff(flag_) == 1))[0]
    if np.ndim(data_) == 1:
        data_ = data_[:, np.newaxis]
    params = np.zeros((data_.shape[1], tds.shape[0]))
    for ii in range(params.shape[1]):
        params[:, ii] = np.mean(data_[tds[ii] + aft_len:tds[ii] + aft_len + avg_len, :], 0) - np.mean(
            data_[tds[ii] - bfm_len - avg_len:tds[ii] - bfm_len, :], 0)
        # print('base', np.mean(data_[tds[ii] + aft_len:tds[ii] + aft_len + avg_len, :], 0))
        # print('peak', np.mean(data_[tds[ii] - bfm_len - avg_len:tds[ii] - bfm_len, :], 0))
    return params.T


if __name__ == '__main__':
    path = '.'
    files = os.listdir(path)
    for f in files:
        if '.txt' in f:
            data = np.genfromtxt(os.path.join(path, f))
            alpha = 3
            beta = 5
            upper = 100
            lower = -100
            step_u = 0
            step_l = 0
            bef = 50
            aft = 50
            avg = 10
            data = preprocess(data)
            data = calc_moving_avg(data, 3)
            energy = calc_energy(data, alpha, beta)
            flag = calc_flag(energy, [upper, lower], [step_u, step_l])
            down_array = np.where(np.diff(flag) == 1)
            up_array = np.where(np.diff(flag) == -1)
            show_fig(data, energy, flag, [upper, lower])
            force = calc_force(data, flag, bef, aft, avg, 100)
