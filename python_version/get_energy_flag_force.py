# encoding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


class DataProcessing:
    def __init__(self, data):
        self.data = data
        self.base = data
        self.force_signal = data
        self.energy = None
        self.flag = None
        self.force = None
        self.mov_avg_len = 5
        self.sigma_wave = 10
        self.sigma_tsunami = 4
        self.alpha = 3
        self.beta = 5
        self.energy_thd = 30
        self.upper = self.energy_thd
        self.lower = self.energy_thd
        self.step_u = 0
        self.step_l = 0
        self.bef = 50
        self.aft = 50
        self.avg = 10

    def pre_process(self):
        if np.ndim(self.data) == 1:
            self.data = self.data - self.data[0]
            self.data = self.data[:, np.newaxis]
        else:
            self.data = self.data - self.data[0, :]
        self.data = -self.data

    def calc_moving_avg(self):
        output = []
        for ii in range(len(self.data)):
            if ii <= self.mov_avg_len:
                output.append(np.mean(self.data[:ii + 1, :], 0))
            else:
                output.append(np.mean(self.data[ii - (self.mov_avg_len - 1):ii + 1, :], 0))
        self.data = np.array(output)

    def baseline_removal_single(self, data):
        base = np.copy(data)
        for ii in range(1, len(data)):
            base[ii] = base[ii - 1] + \
                       (data[ii] - data[ii - 1]) * \
                       np.exp(-(data[ii] - data[ii - 1]) ** 2 / self.sigma_wave) + \
                       (data[ii - 1] - base[ii - 1]) * \
                       np.exp(-np.abs(data[ii - 1] - base[ii - 1]) / self.sigma_tsunami)
        return base

    def baseline_removal(self):
        self.base = np.apply_along_axis(
            lambda x: self.baseline_removal_single(x), 0, self.data
        )
        self.force_signal = self.data - self.base

    def calc_energy(self):
        self.force_signal = np.array(self.force_signal)
        if self.force_signal.ndim == 1:
            self.force_signal = self.force_signal[:, np.newaxis]
        m = self.force_signal.shape[0]
        energy_n = np.zeros((m, 1))
        for ii in range(m - self.alpha - self.beta):
            energy_n[ii + self.alpha + self.beta] = \
                1 / self.alpha * \
                np.sum(np.sum(np.abs(
                    self.force_signal[ii + self.beta:ii + self.beta + self.alpha, :] -
                    self.force_signal[ii:ii + self.alpha, :]), 1), 0)
            # diff_mat = self.force_signal[ii + self.beta:ii + self.beta + self.alpha, :] - self.force_signal[ii:ii + self.alpha, :]
            # diff_mat_max_sub = np.array(np.where(np.abs(diff_mat) == np.max(np.abs(diff_mat))))
            # diff_mat_max_sub = diff_mat_max_sub[:, 0]
            # max_sign = np.sign(diff_mat[diff_mat_max_sub[0], diff_mat_max_sub[1]])
            # energy_n[ii + self.alpha + self.beta] *= max_sign
        self.energy = energy_n.T[0]

    # t is tvalue < thd
    @staticmethod
    def update_flag_status(f, r, t):
        f = f ^ r and f or not (f ^ r) and not (f ^ t)
        r = not r and f and t or r and not (not f and t)
        return f, r

    def calc_flag(self):
        self.flag = np.zeros(self.energy.shape, dtype=np.bool)
        ready = False
        for ii in range(1, self.flag.shape[0]):
            f = bool(self.flag[ii - 1])
            t = (not f and (self.energy[ii] < self.energy_thd)) or (f and (self.energy[ii] < self.energy_thd * 0.9))
            self.flag[ii], ready = self.update_flag_status(f, ready, t)
        self.flag = np.array(self.flag, dtype=np.int)

    def show_fig(self, f):
        fig = plt.figure()
        fig.set_size_inches(60, 10)
        plt.plot(self.force_signal, '-', linewidth=3)
        plt.plot(self.energy)
        plt.hlines(self.upper, 0, self.data.shape[0], linestyles='--')
        plt.hlines(self.lower, 0, self.data.shape[0], linestyles='--')
        plt.plot(self.flag * (np.max(self.data) - np.min(self.data)) + np.min(self.data), '--')
        plt.title(f)
        plt.legend(['rawdata1', 'rawdata2', 'energy', 'upper limit', 'lower limit', 'touch flag'])
        plt.show()

    def calc_force(self):
        tds = np.array(np.where(np.diff(self.flag) == 1))[0]
        if np.ndim(self.force_signal) == 1:
            self.force_signal = self.force_signal[:, np.newaxis]
        params = np.zeros((self.force_signal.shape[1], tds.shape[0]))
        for ii in range(params.shape[1]):
            params[:, ii] = np.mean(self.force_signal[tds[ii] + self.aft:tds[ii] + self.aft + self.avg, :], 0) - \
                            np.mean(self.force_signal[tds[ii] - self.bef - self.avg:tds[ii] - self.bef, :], 0)
        self.force = params.T


if __name__ == '__main__':
    path = '.'
    dirs = os.listdir(path)
    for d in dirs:
        if not os.path.isdir(d):
            continue
        files = os.listdir(os.path.join(path, d))
        for f in files:
            if '.txt' in f:
                data = np.genfromtxt(os.path.join(path, d, f), delimiter=',')
                data = data[:, [0, 4]]
                data = data[:, 0]
                dp = DataProcessing(data)
                dp.pre_process()
                dp.calc_moving_avg()
                dp.baseline_removal()
                plt.plot(dp.data)
                plt.plot(dp.base)
                plt.plot(dp.data - dp.base)
                plt.show()
                dp.calc_energy()
                dp.calc_flag()
                dp.show_fig(f)
                dp.calc_force()
                force = dp.force
                np.savetxt(''.join((f[:-4], '_test.txt')), force, fmt='%8.2f')
