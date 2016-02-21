
clear all
clc

addpath( genpath(pwd) );

Dataset_name = {'AraC'};
% Dataset_name = loadingDatasets;

%=============================================
findoptparam = 0;

cross_num = 3;% k-CV

for iDB = 1:length( Dataset_name )% DB is data base; and length(DB) is number of Data Base.
    
    % load data
    TFBS_name = Dataset_name{ iDB };
    fprintf('Dataset Name:   %s\n', TFBS_name);
    fprintf('Loading   ...');
    load( TFBS_name );% loading dataset
    fprintf('Done\n');
    runtimes = 10; % runing 10 times
    %     fid = fopen( 'results.txt', 'w' );
    % independent run times
    for rn = 1 : runtimes
        % generates cross-validation indices
        indices = kFoldCV( Y, cross_num );% 检验，必须保证每一fold中包含两类
        % indices = crossvalind('Kfold',length(yapp), cross_num);
        %     [train_indx,test_indx] = crossvalind('LeaveMOut',length(yapp));
        bestnfactors = [];
        bestnloadings = [];
        t1 = clock;% count time
        for cros = 1 : cross_num
            fprintf('Starting cross validation: %dth\n', cros );
            %=============== test dataset and train dataset =====================%
            test_indx = (indices == cros);
            train_indx = ~test_indx;
            
            train = X( train_indx, :, : );                     %===================%
            train_label = Y( train_indx );                     %      选择训练集    %
            %                                                  %===================%
            
            test = X( test_indx, :, : );                       %===================%
            test_label = Y(test_indx);                         %     选择测试集     %
            %                                                  %===================%
            
            %=============================================================
            %% find optimial parameters
            fprintf('Finding optimial parameters: %d \n', findoptparam );
            if findoptparam
                nfactors = 30;
                nloading = [ 1 : 30 ];
                [ bestnfactors(cros), bestnloadings(cros) ] = opt_parameters( train, train_label, nfactors, nloading );
                if bestnfactors(cros) == 0
                    bestnfactors(cros) = 11;
                end
                if bestnloadings(cros) == 0
                    bestnloadings(cros) = 5;
                end
            else
                bestnfactors(cros) = 11;
                bestnloadings(cros) = 5;
            end
            %% training
            disp('Training')
            %             model = hoplsc_train( train, train_label, bestnfactors, bestnloadings );
            model = hoplsc_train( train, train_label, bestnfactors(cros), bestnloadings(cros) );
            
            
            %% predicting
            disp('Testing')
            pred = hoplsc_predict( test, model );
            
            
            %% evaluating the performance
            %         results.cv{cros} = assessment( pred.class_pred, test_label );
            %             results.cv(rn,cros) = classperfchr( pred{ bestnfactors }.class_pred, test_label );
            results.cv(rn,cros) = classperfchr( pred{ bestnfactors(cros) }.class_pred, test_label );
            
        end% end for cros
        %% results evaluating
        results.runingtime = etime(clock,t1)/cross_num;
        results.indices_cv{rn} = indices;
        results.bestnfactors{rn} = bestnfactors;
        results.bestnloadings{rn} = bestnloadings;
        clear indices cros train train_label test test_label
    end% end for rn
    
    for rn = 1 : runtimes
        F = 0;
        for cros = 1 : cross_num
            F = F + results.cv(rn,cros).F_measure;
        end
        F_measure( rn ) = F/cross_num;
    end
    results.F_measure = F_measure;
    results.avgF = sum(F_measure)/rn;
    results.stdF = std( F_measure );
    results.numPos = num_class(1);
    save(['res_' 'hopls_' TFBS_name ],  'results');
    results
    clear results
end

