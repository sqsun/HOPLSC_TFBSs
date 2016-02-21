function [r2]=corrcoef4vectwise(Y1,Y2)
%
%  Y1 and Y2 have same # of rows and columns
[r,c] = size(Y1);
r2 = zeros(1,c);
for i=1:c
    temp = corrcoef(Y1(:,i),Y2(:,i));
    r2(i) = temp(1,2);
end