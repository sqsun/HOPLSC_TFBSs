function indices = kFoldCV(yapp,cross_num)
flag = 1;
while flag
%	cross_num = 10;
	indices = crossvalind('Kfold',length(yapp), cross_num);
	for cros = 1:cross_num
		test_indx = (indices == cros);
		test_label = yapp(test_indx);
		if length(unique(test_label)) ~= 2
			flag = 1;
			break;
		else
			flag = 0;
		end
	end
end