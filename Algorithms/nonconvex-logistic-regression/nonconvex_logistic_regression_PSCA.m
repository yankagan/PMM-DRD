function [theta_opt, obj_hist, time_elapsed] = nonconvex_logistic_regression_PSCA(X, y, ...
    theta_0, lambda, eps, n_proc, n_blks_per_proc, options) 
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
% n_proc    The number of processors to use for parallel processing 
%
% n_blks_per_proc The number of blocks processed by each processor in
%           parallel. We assume the block size is 1
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
% This code implements the parallel successive convex approximation (PSCA)
% algorithm. In particular, we implement the cyclic variable selection
% procedure for the parallel block coordinate descent (parallel BCD). Just
% like in the authors' paper, we assume the block size is 1.
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

if ~exist('n_proc', 'var') || isempty(n_proc)
    n_proc = 4;
end

if ~exist('n_blks_per_proc', 'var') || isempty(n_blks_per_proc)
    n_blks_per_proc = 40;
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

% The learning rate at the first iteration
if ~isfield(options, 'learning_rate_start') || isempty(options.learning_rate_start)   
    options.learning_rate_start = 1;
end

% The learning rate exponent. 0.5 < options.learning_rate_exp <= 1 
if ~isfield(options, 'learning_rate_exp') || isempty(options.learning_rate_exp)    
    options.learning_rate_exp = 0.75;
else
    if options.learning_rate_exp <= 0.5
        warning('Learning rate not guaranteed to lead to convergence');
    end
    
    if options.learning_rate_exp > 1
        warning('Learning rate is too small');
    end
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

% Setup the cyclic variable selection schedule
num_rounds = ceil(p/(n_proc * n_blks_per_proc));

cyclic_var_sel = cell(num_rounds, 1);

cycle_offset = n_proc * n_blks_per_proc;
for ix = 1: num_rounds 
    offset = (ix - 1) * cycle_offset;
        
    per_round_var_list = [];
    for iy = 1: n_proc
        start_idx = (iy - 1) * n_blks_per_proc + 1;
        stop_idx = iy * n_blks_per_proc;
        
        start_idx_rd = start_idx + offset;
        stop_idx_rd = stop_idx + offset;
        stop_idx_rd = min(stop_idx_rd, p);
        
        if start_idx_rd <= p
            per_round_var_list = [per_round_var_list; (start_idx_rd: stop_idx_rd)']; %#ok<*AGROW>
        end
    end
    
    cyclic_var_sel{ix} = per_round_var_list;    
end

% Columnize
y = y(:);
theta_k = theta_0(:);

Xy = bsxfun(@times, X, y);

% Choose the proximal coefficient alpha, so that all the diagonals of the
% hessian matrix are (sufficiently) positive
epsilon = 1e-6; % Added for sufficient strong convexity of objective with respect to each variable
alpha = max(lambda/(4 * eps) + epsilon, 0);

%fprintf('The proximal coefficient is %f\n', alpha);

obj_hist = NaN(options.outer_max_iter + 1, 1);

f_fun_orig = @(theta) obj_fun_orig(theta, Xy, lambda, eps);
g_fun_orig = @(theta) grad_fun_orig(theta, Xy, lambda, eps);

obj_hist(1) = f_fun_orig(theta_k);

grad_norm = norm(g_fun_orig(theta_k));

time_elapsed = zeros(options.outer_max_iter, 1);
acc_hist = NaN(options.outer_max_iter, 1);

obj_dec = 10 * options.tol_fun;
step_norm = 10 * options.tol_step;

% Start the matlabpool
% delete(gcp)
% poolobj = parpool(n_proc);
for k = 1: options.outer_max_iter
    tic;
    
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
    
    % Expansion to be carried out around previous theta_k
    l_vec = Xy * theta_k;
    
    % Compute the stepsize gamma_k
    % Ensure sum_k gamma_k = +inf and sum_k gamma_k^2 < +inf by using p-series test
    gamma_k = options.learning_rate_start/(k^options.learning_rate_exp);
    
    round_idx = mod(k - 1, num_rounds) + 1;
        
    % Parallelize across processors here (without parfor)
    
    % Get the list of variables (blks) to be processed by all processors
    var_list = cyclic_var_sel{round_idx};
    
    if ~isempty(var_list)            
        theta_loc = theta_k(var_list); %#ok<*PFBNS>

        Xy_loc = Xy(:, var_list);

        l_offset = bsxfun(@minus, l_vec, bsxfun(@times, theta_loc', Xy_loc));

        % Determine if a variable in the list is the bias term (for
        % regularization purposes)
        is_bias_logical = var_list == p;

        f_fun = @(theta, var_vec) per_variable_fun(theta, var_vec, theta_loc, Xy_loc, l_offset, is_bias_logical, lambda, eps, alpha);
        g_fun = @(theta, var_vec) per_variable_grad(theta, var_vec, theta_loc, Xy_loc, l_offset, is_bias_logical, lambda, eps, alpha);
        h_fun = @(theta, var_vec) per_variable_hess(theta, var_vec, Xy_loc, l_offset, is_bias_logical, lambda, eps, alpha);

        theta_loc_new = trust_region_1D_vectorized(theta_loc, f_fun, g_fun, h_fun, options);

        % Use gamma_k to update theta      
        step_grp = gamma_k * (theta_loc_new - theta_loc);
        theta_grp = theta_loc + step_grp;
        
    else
        theta_grp = [];
    end      
        
    % Update the modified parameters
    theta_k(var_list) = theta_grp;
        
    time_elapsed(k) = toc;
    
    if options.compute_obj
        obj_hist(k + 1) = f_fun_orig(theta_k);
        
        obj_dec = obj_hist(k + 1) - obj_hist(k);
    end
    
    grad_norm = norm(g_fun_orig(theta_k));
    
    step_k = zeros(size(theta_k));
    step_k(var_list) = step_grp;
    
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

% Delete the parallel pool object
% delete(poolobj);

% Trim the results
obj_hist = obj_hist(1: k + 1);
time_elapsed = time_elapsed(1: k);


end 


%**************************************************************************
%                           Helper functions
%**************************************************************************
% Objective function for each variable
function fun_vec = per_variable_fun(theta, var_vec, theta_hat, Xy, l_offset, is_bias_logical, lambda, eps, alpha)
% Assumes theta, var_vec, theta_hat, and is_bias_logical are column vectors
    theta = theta(var_vec);
    theta_hat = theta_hat(var_vec);
    is_bias_logical = is_bias_logical(var_vec);
    Xy = Xy(:, var_vec);
    l_offset = l_offset(:, var_vec);
    
    l_mat = bsxfun(@times, theta', Xy) + l_offset;
        
    f_fun_l_mat = obj_fun_1d(l_mat);
    
    v = log(theta .^ 2 + eps);
    v(is_bias_logical) = 0;   
    
    fun_vec = sum(f_fun_l_mat, 1)' + lambda * v + 0.5 * alpha * (theta - theta_hat).^2;
end

% Gradient for each variable
function grad_vec = per_variable_grad(theta, var_vec, theta_hat, Xy, l_offset, is_bias_logical, lambda, eps, alpha)
% Assumes theta, var_vec, theta_hat, and is_bias_logical are column vectors
    theta = theta(var_vec);
    theta_hat = theta_hat(var_vec);
    is_bias_logical = is_bias_logical(var_vec);
    Xy = Xy(:, var_vec);
    l_offset = l_offset(:, var_vec);
    
    l_mat = bsxfun(@times, theta', Xy) + l_offset;
            
    fder1_fun_l_mat = grad_fun_1d(l_mat);
    
    v = (2 * theta)./(theta .^ 2 + eps);
    v(is_bias_logical) = 0;
    
    grad_vec = sum(fder1_fun_l_mat .* Xy, 1)' + lambda * v + alpha * (theta - theta_hat);    
end

% Hessian for each variable
function hess_vec = per_variable_hess(theta, var_vec, Xy, l_offset, is_bias_logical, lambda, eps, alpha)
% Assumes theta, var_vec, and is_bias_logical are column vectors
    theta = theta(var_vec);
    is_bias_logical = is_bias_logical(var_vec);
    Xy = Xy(:, var_vec);
    l_offset = l_offset(:, var_vec);
    
    l_mat = bsxfun(@times, theta', Xy) + l_offset;
        
    fder2_fun_l_mat = hess_fun_1d(l_mat);
    
    v = (2*(eps - theta .^ 2)) ./ ((theta .^ 2 + eps).^2);
    v(is_bias_logical) = 0;
    
    hess_vec = sum(fder2_fun_l_mat .* (Xy .^ 2), 1)' + lambda * v + alpha;
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
