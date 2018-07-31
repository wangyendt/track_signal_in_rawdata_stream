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
        self.force_signal = np.zeros_like(data)
        self.energy = None
        self.flag = None
        self.force = None
        self.tds = None
        self.limiting_thd = 1000
        self.limiting_step_ratio = 0.4
        self.mov_avg_len = 5
        self.sigma_wave = 10
        self.sigma_tsunami = 4
        self.alpha = 3
        self.beta = 5
        self.energy_thd = 30
        self.min_td_time = 50
        self.min_tu_time = 50
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

    def limiting_filter(self):
        output = np.zeros(np.shape(self.data))
        output[0] = self.data[0]
        for ii in range(len(self.data) - 1):
            for jj in range(np.shape(self.data)[1]):
                if np.abs(self.data[ii + 1, jj] - output[ii, jj]) >= self.limiting_thd:
                    output[ii + 1, jj] = output[ii, jj] + (
                            self.data[ii + 1, jj] - output[ii, jj]) * self.limiting_step_ratio
                else:
                    output[ii + 1, jj] = self.data[ii + 1, jj]
        self.data = output

    def calc_moving_avg(self):
        output = []
        for ii in range(len(self.data)):
            if ii <= self.mov_avg_len:
                output.append(np.mean(self.data[:ii + 1, :], 0))
            else:
                output.append(np.mean(self.data[ii - (self.mov_avg_len - 1):ii + 1, :], 0))
        self.data = np.array(output)

    def baseline_removal_single(self, data):
        # base = peakutils.baseline(data,100)
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
    def update_flag_status(f, r, t, tdf, tuf):
        f_ = f ^ r and f or not (f ^ r) and not (f ^ t)
        f_ = not f and f_ and tuf or f and (f_ or not tdf)
        r_ = not r and f and t or r and not (not f and t)
        return f_, r_

    def calc_flag(self):
        self.flag = np.zeros(self.energy.shape, dtype=np.bool)
        ready = False
        touch_down_frm = 0
        touch_up_frm = self.min_tu_time + 1
        for ii in range(1, self.flag.shape[0]):
            f = bool(self.flag[ii - 1])
            t = (not f and (self.energy[ii] < self.energy_thd)) or (f and (self.energy[ii] < self.energy_thd * 0.9))
            touch_down_frm = touch_down_frm + 1 if self.flag[ii - 1] else 0
            touch_up_frm = touch_up_frm + 1 if not self.flag[ii - 1] else 0
            tdf = touch_down_frm >= self.min_td_time
            tuf = touch_up_frm >= self.min_tu_time
            self.flag[ii], ready = self.update_flag_status(f, ready, t, tdf, tuf)
        self.flag = np.array(self.flag, dtype=np.int)

    def show_fig(self, f):
        plt.rcParams['font.family'] = 'YouYuan'
        fig = plt.figure()
        fig.set_size_inches(60, 10)
        plt.subplot(211)
        plt.plot(self.data)
        plt.title('rawdata')
        plt.legend(tuple([''.join(('rawdata', str(ii))) for ii in range(np.shape(self.data)[1])]))
        plt.ylabel('ADC')
        plt.subplot(212)
        # plt.plot(self.force_signal, '-', linewidth=3)
        plt.plot(self.energy)
        plt.hlines(self.upper, 0, self.data.shape[0], linestyles='--')
        plt.hlines(self.lower, 0, self.data.shape[0], linestyles='--')
        plt.plot(self.flag * (np.max(self.energy) - np.min(self.energy)) + np.min(self.energy), '--')
        plt.title(f)
        plt.xlabel('Time Series')
        plt.ylabel('ADC')
        plt.legend(['energy', 'upper limit', 'lower limit', 'touch flag'])
        plt.show()

    def calc_force(self):
        self.tds = np.array(np.where(np.diff(self.flag) == 1))[0]
        if np.ndim(self.force_signal) == 1:
            self.force_signal = self.force_signal[:, np.newaxis]
        params = np.zeros((self.force_signal.shape[1], self.tds.shape[0]))
        for ii in range(params.shape[1]):
            params[:, ii] = np.mean(self.force_signal[self.tds[ii] + self.aft:self.tds[ii] + self.aft + self.avg, :],
                                    0) - \
                            np.mean(self.force_signal[self.tds[ii] - self.bef - self.avg:self.tds[ii] - self.bef, :], 0)
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
                dp.limiting_filter()
                dp.calc_moving_avg()
                dp.baseline_removal()
                # plt.plot(dp.data)
                # plt.plot(dp.base)
                # plt.plot(dp.data - dp.base)
                # plt.show()
                dp.calc_energy()
                dp.calc_flag()
                dp.show_fig(f)
                dp.calc_force()
                force = dp.force
                np.savetxt(''.join((f[:-4], '_test.txt')), force, fmt='%8.2f')
