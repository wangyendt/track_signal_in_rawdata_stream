function ret = pre_processing(data, win_len)

data = bsxfun(@minus, data, data(1,:));
avg_data = data;
for ii = 1:size(data,1)
    if ii <= win_len
        avg_data(ii,:) = mean(data(1:ii,:),1);
    else
        avg_data(ii,:) = mean(data(ii-win_len+1:ii,:),1);
    end
end
ret = avg_data;

end