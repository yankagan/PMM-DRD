function [acc, y_pred] = linear_classification_accuracy(xTe, yTe, theta)
%   
% Inputs:
% xTe       The test feature matrix. It has dimensions: # of examples x # 
%           of features
%
% yTe       The known labels. 
%
% theta     The classifier parameter vector. The last element of theta is 
%           the bias
%
% Output:
% acc       The classification accuracy (in percent)
%
% y_pred    The predicted labels
%
%
% 05/31/15

theta = theta(:);
w = theta(1: end - 1);
b = theta(end);

y_pred = sign(xTe * w + b);

% Flip a coin for examples on the boundary
undecided_logical = find(y_pred == 0);
if ~isempty(undecided_logical)
    dec = rand(1, numel(undecided_logical)) > 0.5; % dec set to 0 and 1
    dec(dec == 0) = -1; % change 0 to -1
    y_pred(undecided_logical) = dec;
end

acc = 100*sum(y_pred(:) == yTe(:))/numel(y_pred);
