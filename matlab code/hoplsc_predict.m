
% Prediction procedure by HOPLS T2T model (NIPS 2011)
% Input:
%   Xtest: test data with same tensor structure with training data.
%   model: model learned from training data
% Output:
%   Yp:   prediction of Ytest according to Xtest.
%



function pred = hoplsc_predict( Xtest, model )

%% preprocessing
DimX = size( Xtest );
OrdX = length( DimX );

if model.set.prepr 
    model.set.param.reverse = -1;
    [ Xtest, MeansX ] = pretre( Xtest, model.set.param,...
        model.set.MeansX );
end
%%
nfactor = model.nfactor;
Wtpls = model.Wtpls ;

Xnew = double( tenmat( tensor( Xtest ), 1 ) );

for nfac = 1:nfactor
    
    Yp{nfac} = Xnew * Wtpls{ nfac };%
    
    %% post processing for Y
    if model.set.prepr 
        model.set.param.reverse = -1;
        Yp{nfac} = pretre( Yp{nfac}, model.set.param, model.set.MeansY );
    end
    %%
    assigned_class = hoplscfindclass( Yp{nfac}, model.set.thr);
    
    %% results for predict
    pred{nfac}.class_pred = assigned_class';
    pred{nfac}.yc = Yp{nfac};
end

