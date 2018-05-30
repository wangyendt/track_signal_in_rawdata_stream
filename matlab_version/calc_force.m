function [force, tds] = calc_force(data, flag, params)

bef = params(1);
aft = params(2);
avg = params(3);

tds = find(diff(flag)==1);
force = zeros(size(tds));
for ii = 1:length(tds)
    force(ii) = mean(data(tds(ii)+aft:tds(ii)+aft+avg-1,:))- ...
    mean(data(tds(ii)-bef-avg+1:tds(ii)-bef,:));
end

end