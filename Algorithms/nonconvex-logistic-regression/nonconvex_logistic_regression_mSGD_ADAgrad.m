function [theta_opt, obj_hist, time_elapsed] = nonconvex_logistic_regression_mSGD_ADAgrad(X, y, ...
    theta_0, lambda, eps, options)
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
% eps       The positive constant added to theta^2 in the regularizer to
%           ensure log() is bounded below
%
% options   A struct that has fields for the overall algorithm
%
%           The custom settings:
%           options.outer_max_iter = 500
%           options.tol_fun = 1e-9
%           options.tol_grad = 1e-5
%           options.tol_step = 1e-9
%           options.display = 1
%           options.compute_obj = 1
%           options.adagrad_h_init = 1e-2
%           options.adagrad_learning_rate = 1e-2
%           options.mini_batch_size = 100
%
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
%               l(nu) = log(1 + exp(-nu)),
%
% the logistic function.
%
% The regularizer is a nonconvex log penalty log(theta^2 + eps)
%
% The function assumes the bias term is given by the last entry in theta,
% so that the last columnn of X is all ones. Also, the bias term is NOT
% regularized.
%
% This code implements mini-batch stochastic gradient descent with 
% ADAgrad by Duchi et al. 2011
%
% 11/19/15

if nargin < 3
    error('You must specify the first three inputs');
end

% Regularization coefficients
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

% The parameter that sets the lower bound of the penalty term
if ~exist('eps', 'var') || isempty(eps)
    eps = 1e-4;
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

% Whether to compute the objective
if ~isfield(options, 'compute_obj') || isempty(options.compute_obj)    
    options.compute_obj = 1;
end

% The ADAgrad initial gradient history
if ~isfield(options, 'adagrad_h_init') || isempty(options.adagrad_h_init)   
    options.adagrad_h_init = 1e-2;
end

% The ADAgrad learning rate
if ~isfield(options, 'adagrad_learning_rate') || isempty(options.adagrad_learning_rate)    
    options.adagrad_learning_rate = 1e-2;
end

% The mini batch size
if ~isfield(options, 'mini_batch_size') || isempty(options.mini_batch_size)    
    options.mini_batch_size = 100;
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
theta_k = theta_0(:);

Xy = bsxfun(@times, X, y);

obj_hist = NaN(options.outer_max_iter + 1, 1);

f_fun_orig = @(theta) obj_fun_orig(theta, Xy, lambda, eps);
g_fun_orig = @(theta) grad_fun_orig(theta, Xy, lambda, eps); 

obj_hist(1) = f_fun_orig(theta_k);

grad_norm = norm(g_fun_orig(theta_k));

time_elapsed = zeros(options.outer_max_iter, 1);
acc_hist = NaN(options.outer_max_iter, 1); 

obj_dec = 10 * options.tol_fun;
step_norm = 10 * options.tol_step;

grad_hist = options.adagrad_h_init * ones(p, 1);
for k = 1: options.outer_max_iter
    tic;
    
    % Choose a mini-batch uniformly at random
    exp_idx = randperm(n);
    exp_idx = exp_idx(1: options.mini_batch_size);
    
    if abs(obj_dec) < options.tol_fun
        fprintf('Stopping Criterion Met: objective value decrease is lower than threshold\n');
        break;
    end
    
    if grad_norm < options.tol_grad
        fprintf('Stopping Criterion Met: gradient norm is lower than threshold \n');
        break;
    end
    
    if step_norm < options.tol_step
        fprintf('Stopping Criterion Met: step norm is lower than threshold\n');
        break;
    end
        
    % Multiply lambda by the relative size of the example subset
    lambda_effective = lambda * numel(exp_idx)/n;
    
    fg_fun = @(theta) nonconvex_logistic_regression_fun_grad(theta, Xy(exp_idx, :), lambda_effective, eps);
    
    [~, grad_k] = fg_fun(theta_k);   
    
    grad_hist = grad_hist + grad_k.^2;
    
    id_k = grad_k ./ sqrt(grad_hist);        
    
    step_k = - options.adagrad_learning_rate * id_k;
    theta_k = theta_k + step_k;
    
    time_elapsed(k) = toc;
    
    if options.compute_obj
        obj_hist(k + 1) = f_fun_orig(theta_k);
        %obj_hist(k + 1) = fun_new;
        
        obj_dec = obj_hist(k + 1) - obj_hist(k);
    end
    
    grad_norm = norm(g_fun_orig(theta_k));
    
    step_norm = norm(step_k);
        
    if mod(k, options.n_display) == 0
        %if options.display
            fprintf('Iteration: %d/%d, Time taken(s): %f, Obj. Val.: %f, Grad. Norm: %f, Step Norm: %f\n', k, options.outer_max_iter, sum(time_elapsed(k - options.n_display + 1: k)), obj_hist(k), grad_norm, step_norm);
        %end
    end
    
    
    if sum(time_elapsed(1: k)) >= options.max_train_time
        fprintf('Outer Iteration: Maximum training time reached\n');
        break;
    end
end

theta_opt = theta_k;

% Trim the results
obj_hist = obj_hist(1: k + 1);
time_elapsed = time_elapsed(1: k);


end 


%**************************************************************************
%                           Helper functions
%**************************************************************************
function [obj_fun, grad] = nonconvex_logistic_regression_fun_grad(theta, Xy, lambda, eps)
    % Assumes theta is a column vector
    l_vec = Xy * theta;

    obj_fun = sum(obj_fun_1d(l_vec)) + lambda * sum(log(theta(1: end - 1) .^ 2 + eps)); 

    if nargout > 1
        fder1_fun_l_vec = grad_fun_1d(l_vec);

        v = (2 * theta)./(theta .^ 2 + eps);
        v(end) = 0;

        grad = fder1_fun_l_vec' * Xy + lambda * v';

        grad = grad';
    end
end

% Objective function (original)
function obj_fun = obj_fun_orig(theta, Xy, lambda, eps) 
% Assumes theta is a column vector
    l_vec = Xy * theta;
    
    v = log(theta .^ 2 + eps);
    
    v(end) = 0;
        
    obj_fun = sum(obj_fun_1d(l_vec)) + lambda * sum(v);    
end

% Gradient function (original)
function grad = grad_fun_orig(theta, Xy, lambda, eps)
% Assumes theta is a column vector
    l_vec = Xy * theta;
        
    fder1_fun_l_vec = grad_fun_1d(l_vec);

    v = (2 * theta)./(theta .^ 2 + eps);
    
    v(end) = 0;
    
    grad = fder1_fun_l_vec' * Xy + lambda * v';

    grad = grad';
end

% Objective function, data-fit term
function obj_fun = obj_fun_1d(v)
    obj_fun = log(1 + exp(-v));
    
    obj_fun(v > 40) = 0;
    obj_fun(v < -700) = -v(v < -700);
end

% Gradient function, data-fit term
function grad_fun = grad_fun_1d(v)
    exp_neg_v = exp(-v);
    one_plus_exp = 1 + exp_neg_v;
    grad_fun = - exp_neg_v ./ one_plus_exp;
    
    grad_fun(v > 700) = 0;
    grad_fun(v < -700) = -1;
end

% Hessian function, data-fit term
function hess_fun = hess_fun_1d(v) %#ok<*DEFNU>
    exp_neg_v = exp(-v);
    one_plus_exp = 1 + exp_neg_v;
    hess_fun = exp_neg_v ./ (one_plus_exp .^ 2);
    
    hess_fun(v < -700 | v > 700) = 0;
end
