function energy = calc_energy(data, params)

alpha = params(1);
beta = params(2);
gamma = params(3);

m = size(data,1);
energy_n = zeros(m,1);

for ii = 1:m-alpha-beta
    energy_n(ii+alpha+beta) = 1/alpha* ...
        sum(sum(abs(data(ii+beta:ii+beta+alpha,:)-data(ii:ii+alpha,:)),2),1);
    diff_mat = data(ii+beta:ii+beta+alpha,:)-data(ii:ii+alpha,:);
    diff_mat_max_sub = find(abs(diff_mat) == max(abs(diff_mat)));
    diff_mat_max_sub = diff_mat_max_sub(1);
    [diff_mat_max_sub_1, diff_mat_max_sub_2] = ind2sub(size(diff_mat), diff_mat_max_sub);
    max_sign = sign(diff_mat(diff_mat_max_sub_1, diff_mat_max_sub_2));
    energy_n(ii+alpha+beta) = energy_n(ii+alpha+beta) * max_sign;
end
energy = energy_n;

end