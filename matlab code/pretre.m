function [ X, meanX ] = pretre( X1, param, mX )
% pre and postprocessing of multiway arrays, matrix and vector
%
DimX = size( X1 );
Xnew = X1;
reverse = param.reverse;


%% 3D
%
%% 3D
if length( DimX ) > 2% N-way normalization
    [ nXnew, mXnew, kXnew ] = size( Xnew );
    
    if reverse == 1% preprocessing
        Xnew = tensor( Xnew );% covert it into tensor format

        norm_mat = zeros( mXnew, kXnew );
        
        for dim1 = 1:nXnew
            tenXnew = tenmat( Xnew( dim1, :, : ), 1);
            matXnew = tenXnew.data;
            norm_mat =  norm_mat + matXnew ;
        end
        mX = ( norm_mat )./nXnew;% mean
        
        for dim1 = 1:nXnew
            tenXnew = tenmat( Xnew( dim1, :, : ), 1);
            matXnew = tenXnew.data;
            X(dim1,:,:) =  matXnew - mX;
        end
        
        meanX = mX;
        
    elseif reverse == -1% exist mX
        Xnew = tensor( Xnew );
        
        for dim1 = 1:nXnew
            tenXnew = tenmat( Xnew( dim1, :, : ), 1);
            matXnew = tenXnew.data;
            X(dim1,:,:) =  matXnew - mX;
        end
        meanX = mX;
    end

else
    %% for normalizing matrix
    % normalize inputs and output mean and standard deviation to 0 and 1
    tol = 1e-5;
    nbXnew = size( Xnew, 1 ); % number of samples
    if reverse == 1% preprocessing
        
        if ~exist( 'mX' ) 
            meanX = mean( Xnew );
        else
            meanX = mX;
        end;
        
        X = ( Xnew - ones(nbXnew,1)*meanX );
        
    else% post processing
        % centering
        meanX = mX;
        X = ( Xnew - ones(nbXnew,1)*meanX);

    end % end if reverse
end % end if length
