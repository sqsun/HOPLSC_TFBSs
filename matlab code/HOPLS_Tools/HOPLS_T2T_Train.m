 
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

% Developed by Qibin Zhao
% $ Version 1.02 $ May 2010 $
% Version 2.1  Feb 2011 


function [model] = HOPLS_T2T_Train(X,Y,nfactor,xloadnum,yloadnum)

% Initialize the output variables
Gnpls_Out=[];
Wtpls =[];

% Calculation of dimension and order 
DimX = size(X);
OrdX = length(DimX);if OrdX==2&size(X,2)==1;OrdX = 1;end
DimY = size(Y);
OrdY = length(DimY);if OrdY==2&size(Y,2)==1;OrdY = 1;end
n = DimY(1);

% Normalization  
% M_X=mean(double(tenmat(X,1)));
% M_Y=mean(double(tenmat(Y,1)));
% S_X=std(double(tenmat(X,1)));
% S_Y=std(double(tenmat(Y,1)));

% CentX = zeros(1,OrdX);
% CentX(1)  = 1;
% ScalX  = CentX;
% [X,MeansX,ScalesX]=mynprocess(X,CentX,ScalX);
% CentY = zeros(1,OrdY);
% CentY(1)  = 1;
% ScalY  = CentX;
% [Y,MeansY,ScalesY]=mynprocess(Y,CentY,ScalY);

 % Calculation of n-Rank of independent variables
Rank_X = zeros(OrdX,1);
    for i=1:OrdX
        Rank_X(i) = rank(double(tenmat(X,i)));
    end
    Rank_Y = zeros(OrdY,1);
    for i=1:OrdY
        Rank_Y(i) = rank(double(tenmat(Y,i)));
    end

flag =2;
% Parameters setting
if flag==0     
    Xln = ceil(Rank_X(2:end)*xloadnum)';
    Yln = ceil(Rank_Y(2:end)*yloadnum)';
    Xln(Xln==0)=1;
    Yln(Yln==0)=1;
end
if flag==1
    Xln = zeros(1,OrdX-1);
    for i=2:OrdX
        Xln(i-1) = pcanumber(double(tenmat(X,i)),xloadnum);
    end
    Yln = zeros(1,OrdY-1);
    for i=2:OrdY
        Yln(i-1) = pcanumber(double(tenmat(Y,i)),yloadnum);
    end     
end
if flag==2
    Xln = repmat(min(xloadnum,min(Rank_X)),1,OrdX-1);
    Yln = repmat(min(yloadnum,min(Rank_Y)),1,OrdY-1);
end

% 
Xres = tensor(X);
Yres = tensor(Y);
SS_X = sum(X(:).^2);
SS_Y = sum(Y(:).^2);


R2_Y = zeros(1,nfactor);
ssx = zeros(1,nfactor);
ssy = zeros(1,nfactor);
PRESS = zeros(1,nfactor);
RMSEP = zeros(1,nfactor);


T=[]; U=[]; C=[];
b=[]; PP=[]; GG =[];
B =[]; WPLS = []; PPLS =[]; CPLS =[];
Yhat4Res = zeros(DimY);


tdx = tucker_als(Xres, [1 Xln],'init','nvecs');
tdy = tucker_als(Yres, [1 Yln],'init','nvecs');


for f=1:nfactor
    fprintf('\n start factor %d computation... \n',f);
    %% 构建初始的P(1) P(2) ... P(N-1) ** Q(1) Q(2） ... Q(M-1)
    % cross covariance tensor between 3D tensor and tensor
    Z = ttt(Xres,Yres,1,1);% tensor to tensor// tensor to vector
    Sd = tucker_als(Z,[Xln Yln],'init','nvecs');
    
    % similar to c, w in PLS
    P = Sd.U(1:OrdX-1);
    Q = Sd.U(OrdX:end);
    
    tdx.U(2:OrdX) = P;
    tdy.U(2:OrdY) = Q;
    
    %% 通过循环迭代求解最优P(1) P(2) ... P(N-1) ** Q(1) Q(2） ... Q(M-1)
    tmp = -1 * ones(1,length(P));
    tdx = tucker_als_new(Xres, [1 tmp], 'init',tdx.U);% 

   %% tdx为最优解
    % latent vector
    t = tdx.U{1};   
    
    % %     % matrix operation intead of tensor operation
    Pkron = P{end};
    for i= length(P)-1:-1:1
       Pkron = kron(Pkron,P{i}); 
    end
    Qkron = Q{end};
    for i= length(Q)-1:-1:1
       Qkron = kron(Qkron,Q{i}); 
    end
      
    

    % %  calculation of Y core
    tdy.U{1} = t;
    tdy.core = ttm(Yres, tdy.U, 1:OrdY, 't'); 
    
    %  waiting for test
%     if tdy.core(1) <0
%         t =- t;
%         tdy.core = -1*tdy.core;
%         tdx.core = -1*tdx.core; 
%     end
    
    % % % Calculation of X loadings 
    wpls = Pkron * pinv(double(tenmat(tdx.core,1)));
    ppls = double((tenmat(tdx.core,1)*Pkron')');
    
    cpls = double((tenmat(tdy.core,1)*Qkron')');
    
    % store in matrices and cell
    WPLS(:,f) = wpls;
    PPLS(:,f) = ppls;
    CPLS(:,f) = cpls;

    
   
    T(:,f) = t;

    
    % % % deflation of X and Y
    Xhat = T * PPLS';   
    Xres = tensor(X - reshape(Xhat,size(X))); 
%     Xres = Xres - full(tdx); 
    Yhat = T * CPLS';
    Yres = tensor(Y - reshape(Yhat,size(Y)));
%     Yres = Yres - full(tdy);
    
    
% % %  evaluation of fitting data      
    ssx(f) = 1 - (norm(Xres).^2)/SS_X;
    ssy(f) = 1 - (sum(Yres(:).^2))/SS_Y;

    
% % %  evaluation of correlation
    R2_Y(f)= mycorrcoef(Yhat(:),Y(:));
    

    disp(['R2-X:',num2str(ssx(f))]);
    disp(['R2-Y:',num2str(ssy(f))]);
    disp(['Correlation: ', num2str(R2_Y(:,f)')]);
end

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



for i=1:nfactor
    Wstar{i} = WPLS(:,1:i) * inv(PPLS(:,1:i)'*WPLS(:,1:i));     
    Wtpls{i} = Wstar{i} * CPLS(:,1:i)';
end

% for i=1:nfactor
%     Wtpls_norm{i}=( repmat(S_X.^(-1)',1,DimY(2)).*Wtpls{i}.*repmat(S_Y,prod(DimX(2:end)),1));
%     Wtpls_norm{i}=[-M_X*Wtpls_norm{i}; Wtpls_norm{i}];
%     Wtpls_norm{i}(1,:)=Wtpls_norm{i}(1,:)+M_Y;
% end

Ypred = double(tenmat(X,1)) * Wtpls{nfactor};
R2tpls = mycorrcoef(Ypred(:),Y(:));
Y = double(Y);
Ypress = sum((Y(:)-Ypred(:)).^2);
Yrmsep = sqrt(Ypress./n);
YQ2 = 1 - Ypress./sum(Y(:).^2);

disp('=====Training data TPLS====');
disp(['Correlation: ', num2str(R2tpls)]);
disp(['RMSEP: ', num2str(Yrmsep)]);
disp(['R2:',num2str(YQ2)]);


model.Wstar =Wstar;
model.Wtpls =Wtpls;
model.nfactor = nfactor;
model.xloadnum = Xln;
model.yloadnum = Yln;
model.T =T;
model.DimX = DimX;
model.DimY = DimY;
model.Yr2 = R2tpls;
model.YQ2 = YQ2;

%   plot([Y(:),Ypred(:)]);

disp('Training is finished.');




function [f]=normalize_vectwise(F)
%USAGE: [f]=normaliz(F);
% normalize send back a matrix normalized by column
% (i.e., each column vector has a norm of 1)

[ni,nj]=size(F);
v=sqrt(sum(F.^2));
f=F./repmat(v,ni,1);


function [r2,rv]=corrcoef4mat(Y1,Y2)
% USAGE: [r2,rv]=corrcoef4mat2(Y1,Y2];
% Compute 1. the squared coefficient of correlation between matrices
%         2  Rv coefficient
%
%  Y1 and Y2 have same # of rows and columns
y1=Y1(:);y2=Y2(:);
rv=((y1'*y2).^2)./( (y1'*y1)*(y2'*y2));
y1=y1-mean(y1);y2=y2-mean(y2);
r2=((y1'*y2).^2)./( (y1'*y1)*(y2'*y2));

function [r2]=corrcoef4vectwise(Y1,Y2)
%
%  Y1 and Y2 have same # of rows and columns
[r,c] = size(Y1);
r2 = zeros(1,c);
for i=1:c
    temp = corrcoef(Y1(:,i),Y2(:,i));
    r2(i) = temp(1,2);
end


function [r]=mycorrcoef(y1,y2)

r = corrcoef(y1,y2);
r = r(1,2);
