function [T,P,W,Wstar,U,b,C,B_pls,Bpls_star,Xori_rec,...
    Yori_rec,...
    Yhat4Res_rec]=myPLS_svds(X,Y,nfactor)
    %Yjack,R2_X,R2_Y,RESS,PRESS,...,
    %Q2,r2_random,rv_random,...
    %Yjack4Press,

% USAGE: [T,P,W,Wstar,U,B,C,Bpls,Bpls_star,Xhat,...
%          Yhat,Yjack,R2x,R2y,RESSy,PRESSy,Q2,r2y_random,rv_random,...
%          Yhat4Press,Yhat4Ress]=PLS_jack_svds(X,Y,nfact)
% NIPALS version of PLS regression using svds instead of NIPALS per se
%       (faster for large data sets).
% X matrix of predictors, Y: matrix of dependent variables
% nfact=number of latent variables to keep [current default rank(X)-1]
% NB: for large datasets keeping nfact small improves performance a lot!
% GOAL:
% Compute the PLS regression coefficients/decomposition
% X=T*P' Y=T*B*C'=X*Bpls  X and Y being Z-scores
%                          B=diag(b)
%    Y=X*Bpls_star with X being augmented with a col of ones
%                       and Y and X having their original units
%    Yjack is the jackknifed estimation of Y
% T'*T=I (NB normalization <> than SAS)
% W'*W=I
% C is unit normalized,           
% U, P are not normalized 
%  [Notations: see Abdi (2003, 2007, 2010),
%               available from www.utdallas.edu/~herve]
% 
%  Xhat,Yhat: reconstituted matrices from pls 
%    with nfact latent variables (i.e., fixed effect)
%  Yjack: reconstitued Y from jackknife 
%   with nfact latent variables (i.e., random effect)
% R2x, R2y: Proportion of variance of X, Y explained by each latent variable 
% RESSy is the residual sum of squares:
%       RESSy=\sum_{i,k} (y_{i,k} - \hat{y}_{i.k})^2
% PRESSy is the PREDICTED residual sum of squares
%       RESSy=\sum_{i,k} (y_{i,k} - \hat{y}_{-(i.k)})^2
%       where \hat{y}_{-(i.k)} is the value obtained
%       without including y_{i,j} in the analysis
% Q2=1 - PRESSy(n)/(RESSy(n-1))
%   ->  Used to choose # of variable keep factor n if Q2_n > limit
%       rule of thumb:  limit =.05 for # observation<100, 0 otherwise
% r2y_random/rv_random: Vector of r2/rv between Y and Yjack 
%                       for each # of latent variables
% Yhat4Press: array of the nfactor Y Jackknifed matrices used to compute PRESS
% Yhat4Ress : array of the nfactor Y (fixed effect) matrices used to compute RESS
% 
%
% Herve Abdi original version 2003. Modifications: 
%   ->  June 2007  (minimize memory storage)
%   ->  July 2007 Add RESS and PRESS (not optimized for that!)
%   ->  September 2008 add svds instead of standard NIPALS
%                Rewrite jackknife for faster results
%  WARNING:  Computation of RESS and PRESS have not been thoroughly checked
%
%
% References (see also www.utd.edu/~herve)
%  1. Abdi, H. (2010). 
%     Partial least square regression, 
%     projection on latent structure regression, PLS-Regression. 
%     Wiley Interdisciplinary Reviews: Computational Statistics, 
%     2, 97-106.
%  2. Abdi, H. (2007). 
%     Partial least square regression (PLS regression). 
%     In N.J. Salkind (Ed.):  
%     Encyclopedia of Measurement and Statistics. 
%     Thousand Oaks (CA): Sage. pp. 740-744.
%  3. Abdi. H. (2003).
%     Partial least squares regression (PLS-regression). 
%     In M. Lewis-Beck, A. Bryman, T. Futing (Eds):  
%     Encyclopedia for research methods for the social sciences. 
%     Thousand Oaks (CA): Sage. pp. 792-795. 

%

X_ori=X;
Y_ori=Y;
maxfac=rank(X)-1;
if exist('nfactor')~=1;nfactor=maxfac;end
% if nfactor > maxfac;nfactor=maxfac;end
M_X=mean(X);
M_Y=mean(Y);
S_X=std(X);
S_Y=std(Y);
X=zscore(X);
Y=zscore(Y);
[nn,np]=size(X) ;
[n,nq]=size(Y)  ;
if nn~= n;
    error(['Incompatible # of rows for X and Y']);
end
% Precision for convergence
epsilon=eps;
% # of components kept
% Initialistion
% The Y set
U=zeros(n,nfactor);
C=zeros(nq,nfactor);
% The X sets
T=zeros(n,nfactor);
P=zeros(np,nfactor);
W=zeros(np,nfactor);
b=zeros(1,nfactor);
R2_X=zeros(1,nfactor);
R2_Y=zeros(1,nfactor);
RESS=zeros(1,nfactor);
PRESS=zeros(1,nfactor);

RMSEP=zeros(nq,nfactor);
cortu=zeros(1,nfactor);
corpred=zeros(nq,nfactor);

% Yhat4Press is a cube 
% of the jackknifed reconstitution of Y
% Needed to compute PRESS
% -> Current version is not optimized for memory usage
%
Yjack4Press=zeros(n,nq,nfactor);
Yhat4Res_rec=zeros(n,nq,nfactor);

Xres=X;
Yres=Y;
SS_X=sum(sum(X.^2));
SS_Y=sum(sum(Y.^2));
 for l=1:nfactor ; 
 [w,delta1,c]=svds(Xres'*Yres,1);
 t=normaliz(Xres*w);
 u=Yres*c;
 % w=normaliz(Xres'*u);
 %X loadings
 p=Xres'*t;
 % c=normaliz(Yres'*t);
 % b coef
 b_l=u'*t;
 % Store in matrices
 b(l)=b_l;
 P(:,l)=p;
 W(:,l)=w;
 T(:,l)=t;
 U(:,l)=u;
 C(:,l)=c;
 % deflation of X and Y
 Xres=Xres-t*p';
 Yres=Yres-(b(l)*(t*c'));
 R2_X(l)=(t'*t)*(p'*p)./SS_X;
 R2_Y(l)=(t'*t)*(b(l).^2)*(c'*c)./SS_Y;
 Yhat4Res_rec(:,:,l)=(T(:,1:l).*repmat(b(1:l),n,1)*C(:,1:l)').*...
          repmat(S_Y,n,1)+(ones(n,1)*M_Y);
%  Y_pred_rec=Y_pred.*repmat(S_Y,n,1)+(ones(n,1)*M_Y);
 RESS(l)= sum(sum((Y_ori-Yhat4Res_rec(:,:,l)).^2));
 
 cortu(l) = corrcoef4vectwise(t,u);
 RMSEP(:,l)=sqrt(sum((Y_ori-Yhat4Res_rec(:,:,l)).^2)./n)';
 corpred(:,l)=corrcoef4vectwise(Y_ori,Yhat4Res_rec(:,:,l))';
end
%RESS=SS_Y.*(1 - cumsum(R2_Y)); % RESS vi R2_Y
% Yhat=X*B_pls;
X_rec=T*P';
% Y_rec=T*diag(b)*C'
Y_rec=T.*repmat(b,n,1)*C';
% Bring back X and Y to their original units
%
Xori_rec=X_rec.*repmat(S_X,n,1)+(ones(n,1)*M_X);
Yori_rec=Y_rec.*repmat(S_Y,n,1)+(ones(n,1)*M_Y);
%Unscaled_Y_hat=Yhat*diag(S_Y)+(ones(n,1)*M_Y);
% The Wstart weights gives T=X*Wstar
for l=1:nfactor
    Wstar{l}=W(:,1:l)*inv(P(:,1:l)'*W(:,1:l));
    % B_pls=Wstar*diag(b)*C';
    B_pls{l}=Wstar{l}.*repmat(b(1:l),np,1)*C(:,1:l)';
    Bpls_star{l}=( repmat(S_X.^(-1)',1,nq).*B_pls{l}.*repmat(S_Y,np,1));
    Bpls_star{l}=[-M_X*Bpls_star{l};Bpls_star{l}];
    Bpls_star{l}(1,:)=Bpls_star{l}(1,:)+M_Y;
end

% New final compact version for Bpls_star

Y_pred=X_ori*B_pls{nfactor}; 
% Y_pred=[ones(n,1),X_ori]*Bpls_star{nfactor}; 

Yerr=sqrt(sum((Y_ori-Y_pred).^2)./n)';
R2pls=corrcoef4vectwise(Y_ori,Y_pred)';
% Now go for the jackknifed version 
   Yjack=zeros(n,nq);
   disp(['Fixed Model Done. Start Jackknife'])
%   for i=1:n;
%     % if n<10    
%       disp(['Jackniffing row #: ',int2str(i),'/',int2str(n)])
%     % end
%     X4j=X_ori;X4j(i,:)=[];
%     Y4j=Y_ori;Y4j(i,:)=[];
%     [leyhat]=PLS4jack(X4j,Y4j,X_ori(i,:),nfactor);
%     Yjack(i,:)=leyhat(nfactor,:);
%   
%     for l=1:nfactor
%       Yjack4Press(i,:,l)=leyhat(l,:);
%     end
%   end
%    r2_random=zeros(1,nfactor);
%    rv_random=zeros(1,nfactor);
%    for l=1:nfactor;
%     PRESS(l)= sum( sum((Y_ori-Yjack4Press(:,:,l)).^2));
%    [r2_random(l),rv_random(l)]=corrcoef4mat(Y_ori,Yjack4Press(:,:,l));
%    end 
%    Q2=1 -PRESS(1:nfactor-1)./RESS(2:nfactor);
%    % From Tenenhaus (1998), p83, 138 for Q2(1)
%    Q2=[PRESS(1)./(nq*(nn-1)),Q2];

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% *****************************************************************
% % The  Functions here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% normaliz PLS_sup
function [f]=normaliz(F);
%USAGE: [f]=normaliz(F);
% normalize send back a matrix normalized by column
% (i.e., each column vector has a norm of 1)
% Herve Abdi: version June 2007
[ni,nj]=size(F);
v=sqrt(sum(F.^2));
f=F./repmat(v,ni,1);
function z=zscore(x);
% USAGE function z=zscore(x);
% gives back the z-normalization for x
% if X is a matrix Z is normalized by column
% Z-scores are computed with 
% sample standard deviation (i.e. N-1)
% see zscorepop
[ni,nj]=size(x);
m=mean(x);
s=std(x);
un=ones(ni,1);
z=(x-(un*m))./(un*s);
% PLS_sup
function [Yhatsup]=PLS4jack(X,Y,xsup,nfactor)
% USAGE: [Yhatsup]=PLSsup4jack(X,Y,xsup,nfactor)
% PLS regression jackknifed for one supplementary element
% Compute the prediction for one supplementary element
% for 1 to nfactor latent variables
% X active IV matrix, Y active DV matrix
% xsup supplementary IV elements
% Yhatsup nfactor by number of col of Y
%         predicted value corresponding to xsup 
% X=T*P' Y=T*B*C'=X*Bpls  X and Y being Z-scores
%                          B=diag(b)
%    Y=X*Bpls_star with X being augmented with a col of ones
%                       and Y and X having their original units
% T'*T=I (NB normalization <> than SAS)
% W'*W=I
% C is unit normalized,           
% U, P are not normalized 
%  [Notations: see Abdi (2003). & Abdi (2007)
%               available from www.utd.edu/~herve]
% nfact=number of latent variables to keep
% default = rank(X)
% Herve Abdi (2007)
%   


% References (see also www.utd.edu/~herve)
%  1. Abdi (2003).
%  Partial least squares regression (PLS-regression). 
%  In M. Lewis-Beck, A. Bryman, T. Futing (Eds):  
%  Encyclopedia for research methods for the social sciences. 
%  Thousand Oaks (CA): Sage. pp. 792-795. 
%  2. Abdi, H. (2007). 
%  Partial least square regression (PLS regression). 
%  In N.J. Salkind (Ed.):  
%  Encyclopedia of Measurement and Statistics. 
%  Thousand Oaks (CA): Sage. pp. 740-744.
%

X_ori=X;
Y_ori=Y;
if exist('nfactor')~=1;nfactor=rank(X);end
M_X=mean(X);
M_Y=mean(Y);
S_X=std(X);
S_Y=std(Y);
X=zscore(X);
Y=zscore(Y);
[nn,np]=size(X) ;
[n,nq]=size(Y)  ;
if nn~= n;
    error(['Incompatible # of rows for X and Y']);
end
Yhatsup=zeros(nfactor,nq);
% Precision for convergence
epsilon=eps;
% # of components kepts
% Initialisation
% The Y set
U=zeros(n,nfactor);
C=zeros(nq,nfactor);
% The X set
T=zeros(n,nfactor);
P=zeros(np,nfactor);
W=zeros(np,nfactor);
b=zeros(1,nfactor);
Xres=X;
Yres=Y;
 for l=1:nfactor 
[w,delta1,c]=svds(Xres'*Yres,1);
 t=normaliz(Xres*w);
 u=Yres*c;
 %X loadings
 p=Xres'*t;
 % b coef
 b_l=u'*t;
 % Store in matrices
 b(l)=b_l;
 P(:,l)=p;
 W(:,l)=w;
 T(:,l)=t;
 U(:,l)=u;
 C(:,l)=c;
 % deflation of X and Y
 Xres=Xres-t*p';
 Yres=Yres-(b(l)*(t*c'));
%
 Wstar=W*pinv(P'*W);
% B_pls=Wstar*diag(b)*C';
 B_pls=Wstar.*repmat(b,np,1)*C';
% New final compact version for Bpls_star
 Bpls_star=( repmat(S_X.^(-1)',1,nq).*B_pls.*repmat(S_Y,np,1));
 Bpls_star=[-M_X*Bpls_star;Bpls_star];
 Bpls_star(1,:)=Bpls_star(1,:)+M_Y;
 % nsup=size(Xsup,1);
 Yhatsup(l,:)=[1,xsup]*Bpls_star; 
 end
%***********************************************************************
function [r2,rv]=corrcoef4mat(Y1,Y2);
% USAGE: [r2,rv]=corrcoef4mat2(Y1,Y2];
% Compute 1. the squared coefficient of correlation between matrices
%         2  Rv coefficient
%
%  Y1 and Y2 have same # of rows and columns
y1=Y1(:);y2=Y2(:);
rv=((y1'*y2).^2)./( (y1'*y1)*(y2'*y2));
y1=y1-mean(y1);y2=y2-mean(y2);
r2=((y1'*y2).^2)./( (y1'*y1)*(y2'*y2));







