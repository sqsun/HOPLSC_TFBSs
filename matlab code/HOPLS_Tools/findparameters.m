% Parameter selection

function  [bestnfactors,bestnloadings,bestperformance,pergrid] = findparameters(X,Y,nfac,nload)

bestperformance =-inf;
bestnfactors = [];
bestnloadings =[];
pergrid =[];

for iload = 1:size(nload,2)
    [bestnfac, bestper, allper] = loo_hopls(X,Y,nfac,nload(:,iload));
    pergrid(:,iload) = allper';
    if (bestper>=bestperformance || iload==1)
        bestperformance = bestper;
        bestnfactors = bestnfac;
        bestnloadings = nload(:,iload);
    else
        break;
    end
    disp([num2str(iload) 'selection of Ln is finished!!!!!!!']);
    disp(['Best R is ' num2str(bestnfactors)]);
    disp(['Best Ln is ' num2str(bestnloadings)]);
    disp(['Cross-validation performance is ' num2str(bestperformance)]);
end



function  [bestnfac,bestper,allper] = loo_hopls(X,Y,nfactors,loadingnum)
%% Parameters
bestper = 0;
bestnfac = 0;
allper =[];
Xdim = size(X);
Ydim = size(Y);
Nf = 5; 
%% HOPLS
YP = cell(nfactors,1);
indices = crossvalind('Kfold', Xdim(1), Nf);
for i = 1:Nf
    test = (indices == i); train = ~test;
    [model] = HOPLS_T2T_Train(X(train,:,:,:),Y(train,:,:,:),nfactors,loadingnum,loadingnum);
    YPone = HOPLS_T2T_Pred(X(test,:,:,:),model);
    for j=1:nfactors
        YP{j}(test,:,:,:)= squeeze(YPone{j});
    end
    disp(['Cross validation of fold ' num2str(i) ' is finished!!!!!!!']);
    disp(['Ln ' num2str(loadingnum)]);
end

for i=1:nfactors
    out = EvalPred(Y,YP{i});
    out = out.YQ2;
    allper(i) = out;
    if out>bestper || i==1
        bestper = out;
        bestnfac = i;
    end
   
end
    
    
    
    
