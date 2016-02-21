
% HOPLS (High-order partial least squares regression)
% Predicting tensor Y from tensor X (latent vectors are computed stepwise with deflation) (NIPS2011)
%
% Usage:
% INPUT
% X        Multi-way array of independent variables
% Y        Multi-way array of dependent variables
% nfactor      Number of latent vectors
% nloading     Number of loadings for each latent vector
%
% OUTPUT
% model


function [model] = hoplsc_train( X, y_true, nfactor, xloadnum )

prepr = 1;% whether to preprocessing or not

yloadnum = 1;
% Initialize the output variables
Wtpls = [];
%% Encodes class membership in binary form
if size(y_true,2) == 1
    Y = binarize_y( y_true );
end


% Calculation of dimension and order
DimX = size( X );
OrdX = length( DimX );if OrdX==2 & size(X,2)==1; OrdX = 1;end
DimY = size( Y );
OrdY = length( DimY );if OrdY==2 & size(Y,2)==1; OrdY = 1;end
n = DimY( 1 );

%% preprocessing
if prepr

    param.reverse = 1;
    [ X, MeansX ] = pretre( X, param );
    model.set.MeansX = MeansX;
    model.set.param = param;
    
    
    param.reverse = 1;
    [ Y, MeansY ] = pretre( Y, param );
    model.set.MeansY = MeansY;
end
model.set.prepr = prepr;
%%
% Calculation of n-Rank of independent variables
Rank_X = zeros( OrdX, 1 );
for i = 1:OrdX
    Rank_X( i ) = rank( double( tenmat( X, i ) ) );
end
Rank_Y = zeros(OrdY,1);

for i=1:OrdY
    Rank_Y(i) = rank(double(tenmat(Y,i)));
end


Xln = repmat( min(xloadnum,min(Rank_X)), 1, OrdX - 1 );
Yln = repmat( min(yloadnum,min(Rank_Y)), 1, OrdY - 1 );

%
Xres = tensor( X );
% Yres = tensor(Y);
Yres = Y';

SS_X = sum(X(:).^2);
SS_Y = sum(Y(:).^2);


R2_Y = zeros( 1, nfactor );
ssx = zeros( 1, nfactor );
ssy = zeros( 1, nfactor );
PRESS = zeros( 1, nfactor );
RMSEP = zeros( 1, nfactor );


T=[]; U=[]; C=[];
b=[]; PP=[]; GG =[]; B =[];
WPLS = []; PPLS =[]; CPLS = [];
Yhat4Res = zeros( DimY );


tdx = tucker_als(Xres, [1 Xln],'init','nvecs');
% tdy = tucker_als(Yres, [1 Yln],'init','nvecs');

for nfac = 1:nfactor
    %     fprintf('\n start factor %dth computation... \n',nfac);
    
    % cross covariance tensor between 3D tensor and tensor
    %     Z = ttt(Xres,Yres,1,1);% tensor to tensor// tensor to vector
    Z = ttm( Xres, Yres, 1 );% tensor times matrix
    
    Sd = tucker_als( Z, [ Yln Xln ], 'init', 'nvecs' );
    %     fprintf('fitting the cross covariance tensor with %f\n', 1-norm(full(Sd)-S)/norm(S));
    
    % similar to c, w in PLS
    
    Q = Sd.U{1:OrdY-1};
    P = Sd.U(OrdY:end);
    
    tdx.U(2:OrdX) = P;
    %     tdy.U(2:OrdY) = Q;% ++++
    
    %*********************************************************
    %    tmp = -1 * ones( 1, length( P ) );
    % tdx = tucker_als_new(Xres, [1 tmp], 'init',tdx.U);
    U_tmp = ttm( Xres, tdx.U, -1, 't');
    
    %***********************************************************
    t = double( tenmat( U_tmp, 1 ) ) * pinv( double( tenmat( tdx.core, 1 ) ) );
    % latent vector
    %     t = tdx.U{1};
    t = t./norm( t );
    % %     % matrix operation intead of tensor operation
    Pkron = P{end};
    for i= length(P)-1:-1:1
        Pkron = kron(Pkron,P{i});
    end
    % %     Qkron = Q{end};
    % %     for i= length(Q)-1:-1:1
    % %        Qkron = kron(Qkron,Q{i});
    % %     end
    
    
    
    % %  calculation of Y core
    % %     tdy.U{1} = t;
    % %     tdy.core = ttm(Yres, tdy.U, 1:OrdY, 't');
    Dr = ( Yres'*Q )' * t;
    %  waiting for test
    %     if tdy.core(1) <0
    %         t =- t;
    %         tdy.core = -1*tdy.core;
    %         tdx.core = -1*tdx.core;
    %     end
    
    % % % Calculation of X loadings
    wpls = Pkron * pinv(double(tenmat(tdx.core,1)));
    ppls = double((tenmat(tdx.core,1)*Pkron')');
    
    % %     cpls = double((tenmat(tdy.core,1)*Qkron')');
    
    % store in matrices and cell
    WPLS( :, nfac ) = wpls;
    PPLS( :, nfac ) = ppls;
    
    CPLS( nfac, nfac ) = Dr;
    QQ( :, nfac ) = Q;
    
    
    T( :, nfac ) = t;
    
    
    % % % deflation of X and Y
    Xhat = T * PPLS';
    Xres = tensor(X - reshape(Xhat,size(X)));
    %     Xres = Xres - full(tdx);
    % %     Yhat = T * CPLS';
    % %     Yres = tensor(Y - reshape(Yhat,size(Y)));
    %     Yres = Yres - full(tdy);
    Yres = Yres -  ( Dr * t * Q' )';
    
    % % %  evaluation of fitting data
    ssx( nfac ) = 1 - ( norm( Xres ).^2 )/SS_X;
    ssy( nfac ) = 1 - ( sum( Yres(:).^2 ) )/SS_Y;
    
    
    
end% end for f

% % % Validation on training data
% X_rec = T * PPLS';
% Y_rec = T * B * C';
% Xerr = double(tenmat(tensor(X),1)) - X_rec;
% Yerr = Y - Y_rec;
% Xerr = sqrt(sum(Xerr(:).^2)/n);
% Yerr = sqrt(sum(Yerr(:).^2)/n);
% fprintf('X RMSEC is %f.\n',Xerr);
% fprintf('Y RMSEC is %f.\n',Yerr);

% % % predition Y from X
%====================================================================
% 理论上应该对PPLS( :, 1 : i )'*WPLS( :, 1 : i )求逆inv；
% 但是在实际情况中，这个矩阵是不一定可逆的，所以此处利用伪逆pinv进行求解
% 在矩阵可逆的情况下，inv()和pinv()的结果是一样的，但是在矩阵不可逆的情况下结果却差别很大，比如：
% tt =[4.3051   -0.0406
%     4.3051   -0.0406];
% >> inv( tt )
% Warning: Matrix is singular to working precision. 
% > In main at 60 
% ans =[Inf   Inf
%    Inf   Inf];
%>> pinv( tt )
% ans =[ 0.1161    0.1161
%    -0.0011   -0.0011 ];
% 另外一种防止不可逆的情况是给对角线元素加上很小的eps.
% for i = 1:nfactor
%     Wstar{ i } = WPLS( :, 1 : i ) * inv( PPLS( :, 1 : i )'*WPLS( :, 1 : i ) );
%     Wtpls{ i } = Wstar{ i } * CPLS( 1:i, 1:i ) * QQ(:, 1 : i )';
% end
for i = 1:nfactor
    Wstar{ i } = WPLS( :, 1 : i ) * pinv( (PPLS( :, 1 : i )'*WPLS( :, 1 : i ) +  diag(rand(i,1))*eps ) );
    Wtpls{ i } = Wstar{ i } * CPLS( 1:i, 1:i ) * QQ(:, 1 : i )';
end
%====================================================================
% for i=1:nfactor
%     Wtpls_norm{i}=( repmat(S_X.^(-1)',1,DimY(2)).*Wtpls{i}.*repmat(S_Y,prod(DimX(2:end)),1));
%     Wtpls_norm{i}=[-M_X*Wtpls_norm{i}; Wtpls_norm{i}];
%     Wtpls_norm{i}(1,:)=Wtpls_norm{i}(1,:)+M_Y;
% end
%% prediction
model.train_results.F_measure = 0;
% for i = 1:nfactor
if ~iscell(Wtpls)
    disp('wait!')
end
    Ypred = double( tenmat( X, 1 ) ) * Wtpls{ nfactor };% the prediction of training dataset

    %% post processing for Y
    % if prepr
    %     param.reverse = -1;
    %     [ Ypred, MeansY, ScalesY ] = norm_meanstd( Ypred, param, model.set.MeansY, model.set.ScalesY );
    % end
    % Pidx = find( y_true == 1 );
    % Nidx = find( y_true == 2 );
    % Tp = T( 1:length( Pidx ), : );
    % Tn = T( length( Pidx ) + 1 : end, : );
    % tp = Tp( :, 1 );       % Score vectors on the first dimension of the minority class
    % tn = Tn( :, 1 );
    % mean_tp =  mean( tp );  % The mean value of the minority class
    % mean_tn = mean( tn );
    % stp = std( tp );
    % stn = std( tn );
    % q = stp/( stp + stn );
    % b2 = mean_tp - ( mean_tp - mean_tn )*q;  % Bias on the first dimension of score vectors
    % % b = Y'*T(:,1)/(T(:,1)'*T(:,1));
    % % % % b1 = b2*b';  % Get the bias on prediction corr
    % Ypred = Ypred - ones(size(Ypred,1),2)*b2;
    %%
    %============================================
    if abs(min( Ypred(:,1) ) - max( Ypred(:,1) ))<eps | abs(min( Ypred(:,2) ) - max( Ypred(:,2) ))<eps
        disp('wait');
    end
    %=========================================
    resthr = hoplscfindthr( Ypred, y_true );
    
%     assigned_class = hoplscfindclass( Ypred, resthr.class_thr );
    
%     train_results = classperfchr( assigned_class', y_true );
%     if i == 1 || model.train_results.F_measure < train_results.F_measure
%         model.class_calc = assigned_class';
%         model.train_results = train_results;
        model.yc = Ypred;
        model.set.thr = resthr.class_thr;
%     end
% end
%%
model.Wstar = Wstar;
model.Wtpls = Wtpls;
model.nfactor = nfactor;
model.xloadnum = Xln;
model.yloadnum = Yln;
model.T = T;
model.DimX = DimX;
model.DimY = DimY;

%% sub-function
function YY = binarize_y( Y )
% function - encodes class membership in binary form
% G - the total number of classes

unics = unique( Y );
G = length( unics );

YY = zeros(length(Y), G );

for g = 1 : G
    YY( find( Y==unics( g ) ), g ) = 1;
end;
