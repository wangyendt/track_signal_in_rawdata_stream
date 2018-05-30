function show_debug_image(data, energy, flag, params)

du = params(1);
ul = params(2);
plot(data);
title('pre-processed rawdata');
xlabel('time: n frames');
ylabel('signal: ADC');
figure();
xx = 1:size(energy,1);
plot(energy);
hold on;
plot(xx, du*ones(size(xx)), '--', 'linewidth', 2);
plot(xx, ul*ones(size(xx)), '--', 'linewidth', 2);
plot(flag*(max(max(data))-min(min(data)))+min(min(data)), '--', 'linewidth', 2);
title('debug energy relative');
xlabel('time: n frames');
ylabel('whatever');
legend('energy', 'upper of touch down thd', 'lower of touch up thd', 'flag');
end