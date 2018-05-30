function show_debug_image(data, energy, flag, tds, params)

du = params(1);
ul = params(2);
bef = params(3);
aft = params(4);
avg = params(5);
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
for ii = 1:length(tds)
    plot(tds(ii)-bef-avg+1:tds(ii)-bef, flag(tds(ii)-bef-avg+1:tds(ii)-bef)*(max(max(data))-min(min(data)))+min(min(data)), 'ko', 'markersize', 10, 'markerfacecolor', 'k');
    plot(tds(ii)+aft:tds(ii)+aft+avg-1, flag(tds(ii)+aft:tds(ii)+aft+avg-1)*(max(max(data))-min(min(data)))+min(min(data)), 'ko', 'markersize', 10, 'markerfacecolor', 'k');
end
title('debug energy relative');
xlabel('time: n frames');
ylabel('whatever');
legend('energy', 'upper of touch down thd', 'lower of touch up thd', 'flag');
end