function flag = calc_flag(energy, params)

du = params(1);
ul = params(2);

flag = zeros(size(energy));
for ii = 1:numel(flag)
    if ii > 1
        if flag(ii-1) == 0
            if energy(ii) > du
                flag(ii) = 1;
                continue;
            else
                flag(ii) = 0;
            end
        end
        if flag(ii-1) == 1
            if energy(ii) < ul
                flag(ii) = 0;
                continue;
            else
                flag(ii) = 1;
            end
        end
    end
end

end