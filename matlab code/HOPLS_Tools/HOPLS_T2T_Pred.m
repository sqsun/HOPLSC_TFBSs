
% Prediction procedure by HOPLS T2T model (NIPS 2011)
% Input:
%   Xtest: test data with same tensor structure with training data.
%   model: model learned from training data
% Output:
%   Yp:   prediction of Ytest according to Xtest.
%
% Developed by Qibin Zhao 2010/06
% Ver 1.0  2011/03


function [Yp] = HOPLS_T2T_Pred(Xtest,model)


Yp =[];

nfactor = model.nfactor;
Wtpls= model.Wtpls ;

DimY =model.DimY;

for i=1:nfactor
      Xnew = double(tenmat(tensor(Xtest),1));
      Yp{i} = Xnew*Wtpls{i};
      Yp{i} = reshape(Yp{i},[size(Yp{i},1) DimY(2:end)]);
end


disp('Predition is finished');


