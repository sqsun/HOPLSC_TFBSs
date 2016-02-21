
% Calculation of Q^2 from estimated YP and real Y.

function  [results] = EvalPred(Y,Yp,flag)

DimY = size(Y);
OrdY = ndims(Y);

if nargin<3
    flag =0;
else 
    flag =1;
end

if flag
    
    R2_Y = zeros(DimY(2)+1,1);
    Ypress = zeros(DimY(2)+1,1);
    Yrmsep = zeros(DimY(2)+1,1);
    YQ2 = zeros(DimY(2)+1,1);
    
    R2_Y(1:DimY(2))= corrcoef4vectwise(Yp,Y)';
    R2_Y(DimY(2)+1)= corrcoef4vectwise(Yp(:),Y(:));
    Ypress(1:DimY(2)) = sum((Y-Yp).^2)';
    Ypress(DimY(2)+1) = sum((Y(:)-Yp(:)).^2);
    Yrmsep(1:DimY(2)) = sqrt(Ypress(1:DimY(2))./DimY(1));
    Yrmsep(DimY(2)+1) = sqrt(Ypress(DimY(2)+1)./(prod(DimY)));
    YQ2(1:DimY(2)) = 1 - Ypress(1:DimY(2))./sum(Y.^2)';
    YQ2(DimY(2)+1) = 1 - sum((Y(:)-Yp(:)).^2)/sum(Y(:).^2)';   

else  
    R2_Y= corrcoef4vectwise(Yp(:),Y(:));
    Ypress = sum((Y(:)-Yp(:)).^2);
    Yrmsep = sqrt(Ypress./(prod(DimY)));
    YQ2 = 1 - sum((Y(:)-Yp(:)).^2)/sum(Y(:).^2)';
end

results.YQ2 = YQ2;
results.R2_Y = R2_Y;
results.Yrmsep = Yrmsep;
results.Ypress = Ypress;



function [r2]=corrcoef4vectwise(Y1,Y2)
%
%  Y1 and Y2 have same # of rows and columns
[r,c] = size(Y1);
r2 = zeros(1,c);
for i=1:c
    temp = corrcoef(Y1(:,i),Y2(:,i));
    r2(i) = temp(1,2);
end