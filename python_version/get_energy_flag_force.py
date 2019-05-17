# encoding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import functools
import time


def func_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start = time.time()
        r = func(*args, **kw)
        print('%s excute in %.3f s' % (func.__name__, (time.time() - start)))
        return r

    return wrapper


def peak_det(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    python version:
    https://gist.github.com/endolith/250860

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        print('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        print('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


class DataProcessing:
    """
    用于从rawdata中提取force

    Author:   wangye
    Datetime: 2019/5/17 11:17

    example:
    dp = DataProcessing(rawdata, f)
    dp.pre_process()
    dp.limiting_filter()
    dp.calc_moving_avg()
    dp.baseline_removal()
    dp.calc_energy()
    dp.calc_flag()
    dp.show_fig()
    dp.calc_force()
    force = dp.force
    if not os.path.exists('result'):
        os.mkdir('result')
    np.savetxt('result\\' +
               (f[:f.rindex('.')].replace('\\', '_')
                if not dp.simple_file_name
                else f[f.rindex('\\') + 1:f.rindex('.')]
                ) + '_result.txt', force, fmt='%8.2f'
               )
    """

    def __init__(self, data, filename):
        self.data = data
        self.filename = filename
        self.base = data
        self.force_signal = np.zeros_like(data)
        self.energy = None
        self.flag = None
        self.force = None
        self.tds = None
        self.energy_peak = None
        self.energy_valley = None
        self.limiting_thd = 1000
        self.limiting_step_ratio = 0.4
        self.mov_avg_len = 5
        self.sigma_wave = 10
        self.sigma_tsunami = 4
        self.alpha = 3
        self.beta = 5
        self.energy_thd = 30
        self.energy_thd_decay_coef = 0.9
        self.leave_eng_peak_ratio = 0.5
        self.energy_peak_detect_delta = 20
        self.min_td_time = 50
        self.min_tu_time = 50
        self.step_u = 0
        self.step_l = 0
        self.bef = 50
        self.aft = 50
        self.avg = 10
        self.simple_file_name = True

    @func_timer
    def pre_process(self):
        self.data = self.data - self.data[0]
        if np.ndim(self.data) == 1:
            self.data = self.data[:, np.newaxis]
        # self.data = -self.data

    @func_timer
    def limiting_filter(self):
        output = np.zeros_like(self.data)
        output[0] = self.data[0]
        for ii in range(len(self.data) - 1):
            for jj in range(np.shape(self.data)[1]):
                if np.abs(self.data[ii + 1, jj] - output[ii, jj]) >= self.limiting_thd:
                    output[ii + 1, jj] = output[ii, jj] + (
                            self.data[ii + 1, jj] - output[ii, jj]
                    ) * self.limiting_step_ratio
                else:
                    output[ii + 1, jj] = self.data[ii + 1, jj]
        self.data = output

    @func_timer
    def calc_moving_avg(self):
        self.data = np.array([
            self.data[:ii + 1].mean(0) if ii <= self.mov_avg_len else
            self.data[ii - (self.mov_avg_len - 1):ii + 1].mean(0) for ii in range(len(self.data))
        ])

    def _baseline_removal_single(self, data):
        # base = peakutils.baseline(data,100)
        base = np.copy(data)
        for ii in range(1, len(data)):
            base[ii] = base[ii - 1] + \
                       (data[ii] - data[ii - 1]) * \
                       np.exp(-(data[ii] - data[ii - 1]) ** 2 / self.sigma_wave) + \
                       (data[ii - 1] - base[ii - 1]) * \
                       np.exp(-np.abs(data[ii - 1] - base[ii - 1]) / self.sigma_tsunami)
        return base

    @func_timer
    def baseline_removal(self, do_baseline_tracking=False):
        if do_baseline_tracking:
            self.base = np.apply_along_axis(
                lambda x: self._baseline_removal_single(x), 0, self.data
            )
            self.force_signal = self.data - self.base
        else:
            self.force_signal = self.data

    @func_timer
    def calc_energy(self):
        self.force_signal = np.array(self.force_signal)
        if self.force_signal.ndim == 1:
            self.force_signal = self.force_signal[:, np.newaxis]
        m = self.force_signal.shape[0]
        energy_n = np.zeros((m, 1))
        for ii in range(m - self.alpha - self.beta):
            energy_n[ii + self.alpha + self.beta] = \
                1 / self.alpha * \
                np.sum(np.abs(
                    self.force_signal[ii + self.beta:ii + self.beta + self.alpha, :] -
                    self.force_signal[ii:ii + self.alpha, :]
                ))
            # diff_mat = self.force_signal[ii + self.beta:ii + self.beta + self.alpha, :] - self.force_signal[ii:ii + self.alpha, :]
            # diff_mat_max_sub = np.array(np.where(np.abs(diff_mat) == np.max(np.abs(diff_mat))))
            # diff_mat_max_sub = diff_mat_max_sub[:, 0]
            # max_sign = np.sign(diff_mat[diff_mat_max_sub[0], diff_mat_max_sub[1]])
            # energy_n[ii + self.alpha + self.beta] *= max_sign
        self.energy = energy_n.T[0]
        self.energy_peak, self.energy_valley = peak_det(self.energy, self.energy_peak_detect_delta)
        if len(self.energy_peak) > 0 and len(self.energy_peak) * 40 < len(self.data):
            self.energy_thd = min(self.energy_peak[:, 1]) * 0.8

    @staticmethod
    def _update_flag_status(f, r, t, tdf, tuf):
        # t is tvalue < thd
        f_ = f ^ r and f or not (f ^ r) and not (f ^ t)
        f_ = not f and f_ and tuf or f and (f_ or not tdf)
        r_ = not r and f and t or r and not (not f and t)
        return f_, r_

    @func_timer
    def calc_flag(self):
        self.flag = np.zeros(self.energy.shape, dtype=np.bool)
        ready = False
        touch_down_frm = 0
        touch_up_frm = self.min_tu_time + 1
        for ii in range(1, self.flag.shape[0]):
            f = bool(self.flag[ii - 1])
            t = (not f and (self.energy[ii] < self.energy_thd)) or \
                (f and (self.energy[ii] <
                        max(self.energy_thd * self.energy_thd_decay_coef,
                            max(self.energy[ii - touch_down_frm:ii + 1]) * self.leave_eng_peak_ratio
                            )))
            # (f and (self.energy[ii] < self.energy_thd * 1.1)) or \
            touch_down_frm = touch_down_frm + 1 if self.flag[ii - 1] else 0
            touch_up_frm = touch_up_frm + 1 if not self.flag[ii - 1] else 0
            tdf = touch_down_frm >= self.min_td_time
            tuf = touch_up_frm >= self.min_tu_time
            self.flag[ii], ready = self._update_flag_status(f, ready, t, tdf, tuf)
        self.flag = np.array(self.flag, dtype=np.int)

    @func_timer
    def show_fig(self):
        plt.rcParams['font.family'] = 'YouYuan'
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.unicode_minus'] = False
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
        if len(self.energy_peak) > 0:
            plt.plot(self.energy_peak[:, 0], self.energy_peak[:, 1], '.')
        plt.plot(self.flag * (np.max(self.energy) - np.min(self.energy)) + np.min(self.energy), '--')
        plt.hlines(self.energy_thd, 0, self.data.shape[0], linestyles='--')
        plt.title(self.filename)
        plt.xlabel('Time Series')
        plt.ylabel('ADC')
        if len(self.energy_peak) > 0:
            plt.legend(['energy', 'energy peak', 'touch flag', 'energy threshold'])
        else:
            plt.legend(['energy', 'touch flag', 'energy threshold'])
        plt.show()

    @func_timer
    def calc_force(self):
        self.tds = np.array(np.where(np.diff(self.flag) == 1))[0]
        if np.ndim(self.force_signal) == 1:
            self.force_signal = self.force_signal[:, np.newaxis]
        params = np.zeros((self.force_signal.shape[1], self.tds.shape[0]))
        for ii in range(params.shape[1]):
            params[:, ii] = np.mean(
                self.force_signal[self.tds[ii] + self.aft:self.tds[ii] + self.aft + self.avg, :], 0
            ) - np.mean(
                self.force_signal[self.tds[ii] - self.bef - self.avg:self.tds[ii] - self.bef, :], 0
            )
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
                dp = DataProcessing(data, f)
                dp.pre_process()
                dp.limiting_filter()
                dp.calc_moving_avg()
                dp.baseline_removal()
                dp.calc_energy()
                dp.calc_flag()
                dp.show_fig()
                dp.calc_force()
                force = dp.force
                np.savetxt(''.join((f[:-4], '_test.txt')), force, fmt='%8.2f')
