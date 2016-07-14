function [theta_opt, obj_hist, time_elapsed] = nonconvex_svm_SD(X, y, ...
    theta_0, lambda, sc, options, X_te, y_te)
% 
% X         The set of examples. It has dimensions: # of examples (n) by  
%           # of features (p)
%
% y         The set of labels for the examples. An n-vector with elements
%           {-1, 1}
%
% theta_0   The initial parameter estimate. A p-vector
%
% lambda    The regularization coefficient
%
% sc        The scale of the 1D nonconvex loss function 
%
% options   A struct that has fields for the overall algorithm
%
%           The custom settings:
%           options.outer_max_iter = 500
%           options.tol_fun = 1e-9
%           options.tol_grad = 1e-5
%           options.tol_step = 1e-9
%           options.display = 1
%
% X_te      The set of test examples. It has dimensions: 
%           # of examples (n_t) by  # of features (p)
%
% y_te      The set of labels for the test examples. 
%           An n_t-vector with elements {-1, 1}
%
% Output:
% theta_opt An optimal minimum point. A p x 1 vector
%
% obj_hist  The history of objective values for the original function
% 
% time_elapsed      Time taken for each iteration   
%
% The loss function is given as 
%      
%               l(nu) = 1 - tanh(sc * nu),
%
% which approximates the 0-1 loss.
%
% The regularizer is a Tikhonov (L2) regularizer
%
% The function assumes the bias term is given by the last entry in theta,
% so that the last columnn of X is all ones. Also, the bias term is NOT
% regularized.
%
% This code  uses Mark Schmidt's MATLAB C/C++ mex code for SD
%
% 11/19/15

% How often to display objective information
if nargin < 3
    error('You must specify the first three inputs');
end

% Regularization coefficients
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

% Scale parameter for tanh
if ~exist('sc', 'var') || isempty(sc)
    sc = 1;
end

% Setup the parameters for the outer loop iteration
if ~exist('options', 'var') || isempty(options)
    options = struct;
end

% The maximum number of outer iterations
if ~isfield(options, 'outer_max_iter') || isempty(options.outer_max_iter)    
    options.outer_max_iter = 500;
end

% The tolerance used to stop the iteration, when the objective function
% doesn't change a lot
if ~isfield(options, 'tol_fun') || isempty(options.tol_fun)    
    options.tol_fun = 1e-9;
end

% The tolerance used to stop the iteration, when norm of the gradient is 
% close to 0
if ~isfield(options, 'tol_grad') || isempty(options.tol_grad)    
    options.tol_grad = 1e-5;
end

% The tolerance used to stop the iteration, when norm of the step is close 
% to 0
if ~isfield(options, 'tol_step') || isempty(options.tol_step)    
    options.tol_step = 1e-9;
end

% Whether to display messages
if ~isfield(options, 'display') || isempty(options.display)    
    options.display = 1;
end

% How often to display objective information
if ~isfield(options, 'n_display') || isempty(options.n_display)
    options.n_display = 1;
end

% The maximum allowed training time in seconds
if ~isfield(options, 'max_train_time') || isempty(options.max_train_time)
    options.max_train_time = 100;
end

% Number of parameters (including bias)
p = numel(theta_0);

% Number of examples
n = numel(y);

% Verify size of X
[n1, p1] = size(X);

if p1 ~= p || n1 ~= n
    error('Ensure the dimensions of X, y, and theta_0 agree');
end

% Columnize
y = y(:);

Xy = bsxfun(@times, X, y);

opts.Method = 'sd';

opts.max_train_time = options.max_train_time;
opts.n_display = options.n_display;
opts.optTol = options.tol_grad;
opts.progTol = options.tol_fun;
opts.stepTol = options.tol_step;
opts.DerivativeCheck = 'Off';
opts.maxIter = options.outer_max_iter;

if options.display
    opts.Display = 'On';
else
    opts.Display = 'Off';
end

fg_fun = @(theta) nonconvex_svm_fun_grad(theta, Xy, sc, lambda);

if ~exist('X_te', 'var') || isempty(X_te) || ~exist('y_te', 'var') || isempty(y_te)
    X_te = [];
    y_te = [];
end

[theta_opt, ~, ~, output] = minFunc_with_time_and_accuracy(fg_fun, theta_0, opts, X_te, y_te);

if nargout > 1
    obj_hist = output.trace.fval;
    time_elapsed = output.timeElapsed;    
end

end 


%**************************************************************************
%                           Helper functions
%**************************************************************************
function [obj_fun, grad] = nonconvex_svm_fun_grad(theta, Xy, sc, lambda)
% Assumes theta is a column vector
    l_vec = Xy * theta;

    obj_fun = sum(obj_fun_1d(l_vec, sc)) + ...
        0.5 * lambda * sum(theta(1: end - 1) .^ 2);

    if nargout > 1
        fder1_fun_l_vec = grad_fun_1d(l_vec, sc);

        grad = fder1_fun_l_vec' * Xy + lambda * ([theta(1: end - 1); 0])';

        grad = grad';
    end

end

% Objective function (original)
function obj_fun = obj_fun_orig(theta, Xy, sc, lambda)
% Assumes theta is a column vector
    l_vec = Xy * theta;
    
    obj_fun = sum(obj_fun_1d(l_vec, sc)) + ...
        0.5 * lambda * sum(theta(1: end - 1) .^ 2);
end

% Gradient function (original)
function grad = grad_fun_orig(theta, Xy, sc, lambda)
% Assumes theta is a column vector
    l_vec = Xy * theta;
    
    fder1_fun_l_vec = grad_fun_1d(l_vec, sc);
    
    grad = fder1_fun_l_vec' * Xy + lambda * ([theta(1: end - 1); 0])';
    
    grad = grad';
end

% Objective function (1D)
function obj_fun = obj_fun_1d(v, sc)
    obj_fun = 1 - tanh(sc * v);
end

% Gradient function (1D)
function grad = grad_fun_1d(v, sc)
    grad = -sc * (sech(sc * v)).^2;
end

% Hessian function (1D)
function hess = hess_fun_1d(v, sc) %#ok<*DEFNU>
    hess = 2 * sc^2 * (sech(sc * v)).^2 .* tanh(sc * v);
end
