


function  [bestnfac,bestper,allper] = loo_npls(X,Y,nfactors)
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
        
    [Xfactors,Yfactors,Core,B]=npls(X(train,:,:,:),Y(train,:,:,:),nfactors);
   
             
    for j=1:nfactors
        YP{j}(test,:)= npred(X(test,:,:,:),j,Xfactors,Yfactors,Core,B);
    end
    disp(['Cross validation of fold ' num2str(i) ' is finished!!!!!!!']);
end

for i=1:nfactors
    out = EvalPred(Y,YP{i});
    out = out.YQ2;
    allper(i) =out;
    if out>bestper || i==1
        bestper = out;
        bestnfac = i;
    end
    disp('===========================================================================');
    disp(['The best number of latent is :', num2str(bestnfac)]);
end