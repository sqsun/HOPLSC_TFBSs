function results = classperfchr( predictions, targets )

% Classification performance characteristics
%
% Inputs:
%   predictions               - Predictions (1xN vector,
%                                   where N - the number of patterns)
%   targets                     - Targets (1xN vector)
%
% Outputs:
%   err                     - Error rate (not in percent!)
%   TPR                         - True positive rate (not in per cent!)
%   FPR                     - False positive rate (not in per
%                                   cent!)
%   TP_FP_spread             - TP-FP spread (not in per cent!)
%   F_measure                   - F_measure (not in per cent!)


% Compute the error rate
err = class_error(predictions, targets);
% err = err/100;

% Compute the confusion matrix
%               Predictions
%                  P    N
%               ***********
% True       P  * TP * FN *
% labels     N  * FP * TN *
%               ***********
CM = confusion_matrix(predictions, targets); % ¼ÆËã»ìÏý¾ØÕó£¬confusion_matrix
% Get the number of true positives
TP = CM(1,1);
% Get the number of true negatives
TN = CM(2,2);
% Get the number of false positives
FP = CM(2,1);
% Get the number of false negatives
FN = CM(1,2);

% Compute the true positive rate (sensitivity) or the ratio of the
% number of true positives to the number of positives
TPR = TP/(TP + FN);
% Compute the false positive rate (1 - specificity) or the ratio of the
% number of false positives to the number of negatives
% Specificity: TN/(FP + TN)
FPR = FP/(FP + TN);
% Compute the TP - FP spread
% It denotes the ability of a classifier to make correct predictions
% while minimizing false alarms
TP_FP_spread = TPR - FPR;
% Compute the F-measure combining TP and FP
% For this we need to compute precision and recall
% Compute precision
Precision = TP/(TP + FP);
% Compute recall
Recall = TPR;
% Finally, compute the F-measure
F_measure = (2*Precision*Recall)/(Precision + Recall);

if isnan(F_measure)
    results.F_measure = 0;
else
    results.F_measure = F_measure;
end

results.err = err;
results.Precision = Precision;
results.Recall = Recall;
results.confusion_matrix = CM;
return

function CM = confusion_matrix(predictions, targets) 
  
% Confusion matrix 
% Inputs: 
%   predictions             - Predictions 
%   targets                 - True labels 
% 
% Output: 
%   CM                      - confusion matrix
% Check if sizes of two vectors are equal 
if ~isequal(size(predictions),size(targets))      
    error('Predictions and targets must have equal length!'); 
end 
  
% Get the number of classes 
Uc = unique(targets); 
c = length(Uc); 
  
% Allocate memory 
CM = zeros(c); 
% Construct the confusion matrix
for i = 1:c 
    for j = 1:c 
        CM(i,j) = length(find(targets == Uc(i) & predictions == Uc(j)));
    end 
end 
% Elements of CM are not per cents!!! 
  
function errprcnt = class_error(test_targets,targets) 
test_targets = test_targets(:);
targets = targets(:);
% Calculate error percentage based on true and predicted test labels 
errprcnt = mean(test_targets ~= targets); 
% errprcnt = 100*errprcnt; 
  