


function  [bestnfac,bestper,allper] = loo_pls(X,Y,nfactors)
%% Parameters
bestper = 0;
bestnfac = [];
allper =[];
Xdim = size(X);
Ydim = size(Y);
Nf = 5; 
%% unfold pls
YP = cell(nfactors,1);
indices = crossvalind('Kfold', Xdim(1), Nf);
for i = 1:Nf
    test = (indices == i); train = ~test;
        
    [T,P,W,Wstar,U,b,C,B_pls,Bpls_star,Xori_rec,...
        Yori_rec,Yhat4Res_rec]=myPLS_svds(X(train,:),Y(train,:),nfactors);
             
    for j=1:nfactors
        YP{j}(test,:)= [ones(size(X(test,:),1),1),X(test,:)]*Bpls_star{j};
    end
    disp(['Cross validation of fold ' num2str(i) ' is finished!!!!!!!']);
end

for i=1:nfactors% 找出误差最小(最好的)的成分(因子factor)
    out = EvalPred(Y,YP{i});
    out = out.YQ2;
    allper(i) = out;
    if out>bestper || i==1
        bestper = out;
        bestnfac = i;
    end
end