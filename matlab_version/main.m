clearvars; close all; clc;
files = dir('*.txt');
for ii = 1:length(files)
    data = load(files(ii).name);
    data = pre_processing(data, 5);
    alpha = 3; beta = 5; gamma = 1;
    du = 3; ul = -3;
    bef = 30; aft = 30; avg = 5;
    energy = calc_energy(data, [alpha, beta, gamma]);
    flag = calc_flag(energy, [du, ul]);
    [force, tds] = calc_force(data, flag, [bef, aft, avg]);
    show_debug_image(data, energy, flag, tds, [du, ul, bef, aft, avg]);
end
