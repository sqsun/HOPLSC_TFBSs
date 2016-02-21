% Parameter selection

function  [bestnfactors,bestnloadings,bestperformance,pergrid] = opt_parameters( X, Y, nfac, nload )

bestperformance = -inf;
bestnfactors = [];
bestnloadings =[];
pergrid = [];

for iload = 1:size( nload, 2 )
    [ bestnfac, bestper, allper ] = loo_hoplsc( X, Y, nfac, nload(:,iload) );
    if bestnfac == 0
        disp('wait')
    end
    pergrid( :, iload ) = allper';
    if ( bestper >= bestperformance || iload == 1 )
        bestperformance = bestper;
        bestnfactors = bestnfac;
        bestnloadings = nload( :, iload );
    else
        break;
    end
%     disp([num2str(iload) 'selection of Ln is finished!!!!!!!']);
%     disp(['Best R is ' num2str(bestnfactors)]);
%     disp(['Best Ln is ' num2str(bestnloadings)]);
%     disp(['Cross-validation performance is ' num2str(bestperformance)]);
end



function  [bestnfac,bestper,allper] = loo_hoplsc( X, Y, nfactors, loadingnum )
%% Parameters
bestper = 0;
bestnfac = 0;
allper = [];

cross_num = 3;% k-CV 
%% HOPLS

% indices = crossvalind('Kfold', Xdim(1), cross_num );
% generates cross-validation indices
indices = kFoldCV( Y, cross_num );% 检验，必须保证每一fold中包含两类
for i = 1 : cross_num% 3-cv交叉验证
    test = (indices == i); 
    train = ~test;
    
    [model] = hoplsc_train( X(train,:,:),Y(train,:),nfactors,loadingnum );
    pred = hoplsc_predict( X(test,:,:), model );
end



for nfac = 1:nfactors
    out =  classperfchr( pred{nfac}.class_pred, Y(test,:) );
    F_measure = out.F_measure;
    allper(nfac) = F_measure;% 当无法正确判断类标F_measure 一直为零
    if F_measure > bestper || i==1
        bestper = F_measure;
        bestnfac = nfac;
    end
   
end
    
    
    
    
