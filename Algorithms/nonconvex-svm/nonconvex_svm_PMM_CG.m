function [theta_opt, obj_hist, time_elapsed] = ...
    nonconvex_svm_PMM_CG_mex(X, y, theta_0, lambda, sc, num_grp, ... 
    param_ord, c_f, i_f, options)  
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
% num_grp   The number of parameter groups (partitions) to use, for
%           coarse parallelization.
%
% param_ord The ordering of the parameters, before contiguous splitting 
%           into num_grp groups
%
% c_f       The curvature factor. 0 <= c_f < 1. This factor will be
%           used to multiply f2der_min to set up a threshold above
%           which l_i's with negative second derivative will have a
%           designed convexity region (CR). This CR has been defined to
%           deal with slow update issues that arise at points with 
%           negative and large curvature (close to 0) relative to the
%           minimum curvature point
%
% i_f       The interval factor.  0 < i_f <= 1. This factor defines how
%           far from the minimum curvature point(s) we want the lower
%           and upper bounds defining the CR to be
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
%           The total number of threads to use in parallel regions of the
%           cpp mex code
%           options.num_threads = min(num_grp, 32);
%
%
% Output:
% theta_opt An optimal minimum point. A p x 1 vector
%
% obj_hist  The history of objective values for the original function
%
% time_elapsed      Time taken for each iteration   
%
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
% NOTE: This version of the code convexifies and lifts for parameter
% splitting, but NOT example splitting.
%
% 11/19/15
%
% This version of the code uses cpp mex to accelerate the parallel for-loop
% over different parameter groups

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

% The number of parameter groups
% Also the number of parallel processors
if ~exist('num_grp', 'var') || isempty(num_grp)
   num_grp = 4;
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

% The total number of threads to use in parallel regions of the
% cpp mex code
if ~isfield(options, 'num_outer_threads') || isempty(options.num_outer_threads)
    options.num_outer_threads = min(num_grp, 32);
end

% The total number of threads to use in the inside parallel regions of the
% cpp mex code
if ~isfield(options, 'num_inner_threads') || isempty(options.num_inner_threads)
    options.num_inner_threads = 0;
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

% Number of groups of parameters
if ~exist('num_grp', 'var') || isempty(num_grp)
    % We assume a group size of 1000 is manageable
    mgs = 1000;
    num_grp = ceil(p/mgs);
end

% Verify size of X
[n1, p1] = size(X);

if p1 ~= p || n1 ~= n
    error('Ensure the dimensions of X, y, and theta_0 agree');
end

% The order of parameters prior to splitting
if ~exist('param_ord', 'var') || isempty(param_ord)
    param_ord = 1: p;
end

% Check the size of param_ord
if numel(param_ord) ~= p
    error('Check the dimension of param_ord');
end

if num_grp > p || num_grp < 1
    error('Verify that the number of groups lies between 1 and p');
end



% Columnize
y = y(:);
theta_k = theta_0(:);

Xy = bsxfun(@times, X, y);

% 09/20/15
% Re-order the parameter dimension using param_ord
% X = X(:, param_ord);
Xy = Xy(:, param_ord);
theta_k = theta_k(param_ord);

% Find the examples that are zero vectors. Compute their contribution to
% the objective, then remove them from X, Xy, and y
afp_ones = sum(abs(Xy), 2); % n x 1
non_zero_exp = afp_ones ~= 0;
n_zero_exp = n - sum(non_zero_exp);

% Contribution of the removed zero example feature vectors to the objective
obj_zero_exp = n_zero_exp;

% Remove the examples from X, Xy, and y
% X = X(non_zero_exp, :);
% y = y(non_zero_exp);
Xy = Xy(non_zero_exp, :);

% Determine the contiguous parameter split points
max_num_param_per_grp = ceil(p/num_grp);
grp_splits = max_num_param_per_grp: max_num_param_per_grp: p;
if max_num_param_per_grp * num_grp ~= p
    grp_splits(end + 1) = p;
end
grp_splits = [0, grp_splits];

% Compute the inverse of the additive convexity constants 1/rho
% Split the feature matrix and label times feature matrix based on the groups
% For each group maintain a list of "good" examples. Those that are not zero vectors
Xy_grp = cell(num_grp, 1);
% X_grp = cell(num_grp, 1);
inv_rho_grp = NaN(n, num_grp);

for g = 1: num_grp
    loc_idx = grp_splits(g) + 1: grp_splits(g + 1);
    param_set = param_ord(loc_idx);
    
    % Locate the bias term
    location = find(param_set == p);    
    if ~isempty(location)        
        % First element is the group index
        % Second element is the location within the group
        bias_location = [g, location];
    end        
    
    % X_grp_loc = X(:, param_set);    
    % X_grp{g} = X_grp_loc;
    
    % Xy has already been re-ordered to match param_ord
    Xy_grp_loc = Xy(:, loc_idx); 
    Xy_grp{g} = Xy_grp_loc;
     
    % L1 norm of the group feature matrix over the parameter set
    %afp_ones_local = sum(abs(X_grp_loc), 2); % n x 1
    afp_ones_local = sum(abs(Xy_grp_loc), 2); % n x 1
     
    % Convexity constant to guarantee convergence of parallelized code
    inv_rho_loc = afp_ones ./ afp_ones_local; 
        
    % Find degenerate case
    % When a group uses a subset of an example, which happens to be a zero
    % vector subset, afp_ones_local will be 0, but afp_ones will be
    % positive, giving Inf    
    % bad_exp_loc_logical = isinf(inv_rho_loc);
    % good_exp_grp{g} = ~bad_exp_loc_logical;    
    
    inv_rho_loc(isinf(inv_rho_loc)) = 0;
    
    inv_rho_grp(:, g) = inv_rho_loc(:);
end

% Indices for obtaining the original parameter vector ordering from the
% concatenated groups of parameters
[~, inv_param_ord] = sort(param_ord);
% [~, inv_param_ord] = unique(param_ord);

% 1D second derivative function and its minimum value
l_min = -atanh(sqrt(3)/3)/sc; 
f2der_fun = @(x) hess_fun_1d(x, sc);
f2der_min = f2der_fun(l_min); % f2der_min = -4*sqrt(3)/9 * sc^2;

% Location(s) where the second derivative achieves min
gmp_vec = l_min;

% Location(s) where the second derivative is zero
inflex_pts = 0;

obj_hist = NaN(options.outer_max_iter + 1, 1);
% obj_hist_convex = NaN(options.outer_max_iter + 1, 1);

% Function handle for non-convex objective and gradient
bias_term = find(param_ord == p); % 09/20/15
f_fun_orig = @(theta) obj_fun_orig(theta, Xy, sc, lambda, bias_term);
g_fun_orig = @(theta) grad_fun_orig(theta, Xy, sc, lambda, bias_term);

obj_hist(1) = f_fun_orig(theta_k);
% obj_hist_convex(1) = obj_hist(1);

grad_norm = norm(g_fun_orig(theta_k));

% Special handling for index of group and location within the group for the
% bias term. -1 for C/C++ indexing
bias_grp = bias_location(1) - 1;
bias_idx = bias_location(2) - 1;

l_vec = Xy * theta_k;

time_elapsed = zeros(options.outer_max_iter, 1);
acc_hist = NaN(options.outer_max_iter, 1);

obj_dec = 10 * options.tol_fun;
step_norm = 10 * options.tol_step;

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
        
    [intervals, d_vec] = ...
        get_lower_upper_proj_bounds(l_vec, gmp_vec, f2der_min, f2der_fun, c_f, i_f, inflex_pts);
    
    % Get the indices of the left and right bounds that are not infinite
    not_inf_idx_l = ~isinf(intervals(:, 1));
    not_inf_idx_r = ~isinf(intervals(:, 2));
    
    % Function handle for convex objective using all the parameters, for
    % keeping track of the overall objective
    % bias_term = find(param_ord == p);
    % f_fun_convex = @(theta) obj_fun_conv(theta, Xy, theta_k, sc, d_vec, lambda, bias_term);
    
    A = [Xy(not_inf_idx_r, :); -Xy(not_inf_idx_l, :)];
    b = [intervals(not_inf_idx_r, 2); -intervals(not_inf_idx_l, 1)];
    
    % Create a matrix with 8 columns containing the lower endpoint,
    % upper endpoint, function value at both endpoints, first
    % derivative at both endpoints, and second derivative at both
    % endpoints of the convexified 1D functions
    fun_grad_hess = zeros(n, 6);
    fun_grad_hess(not_inf_idx_l, 1) = obj_fun_convex_1d(intervals(not_inf_idx_l, 1), l_vec(not_inf_idx_l), d_vec(not_inf_idx_l), sc);
    fun_grad_hess(not_inf_idx_r, 2) = obj_fun_convex_1d(intervals(not_inf_idx_r, 1), l_vec(not_inf_idx_r), d_vec(not_inf_idx_r), sc);
    fun_grad_hess(not_inf_idx_l, 3) = grad_fun_convex_1d(intervals(not_inf_idx_l, 1), l_vec(not_inf_idx_l), d_vec(not_inf_idx_l), sc);
    fun_grad_hess(not_inf_idx_r, 4) = grad_fun_convex_1d(intervals(not_inf_idx_r, 1), l_vec(not_inf_idx_r), d_vec(not_inf_idx_r), sc);
    fun_grad_hess(not_inf_idx_l, 5) = hess_fun_convex_1d(intervals(not_inf_idx_l, 1), d_vec(not_inf_idx_l), sc);
    fun_grad_hess(not_inf_idx_r, 6) = hess_fun_convex_1d(intervals(not_inf_idx_r, 1), d_vec(not_inf_idx_r), sc);    
    
    cr_end_pt_info = [intervals, fun_grad_hess];
    
    if options.display
        % res_k = b - A * theta_k;
        % rc_k = min(res_k ./ sqrt(sum(A .^ 2, 2)));    
        % fprintf('The ball radius is %f before parallelizing\n', rc_k);
    end
        
    % Parallelize over parameter groups within mex function
    theta_new = PMM_CG_nonconvex_svm(Xy_grp, theta_k, grp_splits, ...
        cr_end_pt_info, l_vec, d_vec, lambda, inv_rho_grp, bias_grp, ...
        bias_idx, sc, options, options.num_outer_threads, ...
        options.num_inner_threads);
    
    % Perform the necessary projection and line search
    s_k = theta_new - theta_k;    
    alpha_k = convex_polyhedron_line_segment_intersect(theta_k, s_k, A, b);
    
    if options.display
        fprintf('alpha_k = %f\n', alpha_k);
    end 
    
    step_k = alpha_k * s_k;
    theta_k = theta_k + step_k;      
        
    % Update l_vec   
    l_vec = Xy * theta_k;
    
    time_elapsed(k) = toc;
    
    % Keep track of function and gradient values
    if options.compute_obj
        obj_hist(k + 1) = f_fun_orig(theta_k) + obj_zero_exp;
        % obj_hist_convex(k + 1) = f_fun_convex(theta_k) + obj_zero_exp;
        obj_dec = obj_hist(k + 1) - obj_hist(k);
    end
    
    grad_norm = norm(g_fun_orig(theta_k));
    
    step_norm = norm(step_k);
        
    if mod(k, options.n_display) == 0
        %if options.display
            fprintf('Iteration: %d, Time taken(s): %f, Obj. Val.: %f, Grad. Norm: %f, Step Norm: %f\n', k, sum(time_elapsed(k - options.n_display + 1: k)), obj_hist(k), grad_norm, step_norm);
        %end
    end
    
    if sum(time_elapsed(1: k)) >= options.max_train_time
        fprintf('Outer Iteration: Maximum training time reached\n');
        break;
    end
end

theta_opt = theta_k;

% 09/20/15
% Restore the order of the solution
theta_opt = theta_opt(inv_param_ord);

% Trim the results
obj_hist = obj_hist(1: k + 1);
time_elapsed = time_elapsed(1: k);


end


%**************************************************************************
%                           Helper functions
%**************************************************************************
% Objective function (convexified, split parameters)
function obj_fun = obj_fun_param_split(theta, Xy, theta_hat, sc, ...
    d_vec, lambda, cr_end_pt_info, bias_term, l_vec_old, inv_rho_loc) %#ok<*DEFNU>
% Assumes theta, theta_hat, d_vec, l_vec_old, and inv_rho_loc are column
% vectors
    l_vec = Xy * theta;
    l_vec_hat = Xy * theta_hat;
    
    rho_loc = 1./inv_rho_loc;
    
    rho_loc(isinf(rho_loc)) = 0;
    
    %l_vec_grp = ((l_vec - l_vec_hat) .* inv_rho_loc) + l_vec_old;
    
    l_vec_grp = ((l_vec - l_vec_hat) .* inv_rho_loc);
    l_vec_grp(isnan(l_vec_grp)) = 0;
    l_vec_grp = l_vec_grp + l_vec_old;
    
    % Location of endpoints of convex region
    l_lb = cr_end_pt_info(:, 1);
    l_ub = cr_end_pt_info(:, 2);    
    f_con_lb = cr_end_pt_info(:, 3);
    f_con_ub = cr_end_pt_info(:, 4);
    f_con_der_lb = cr_end_pt_info(:, 5);
    f_con_der_ub = cr_end_pt_info(:, 6);
    f_con_2der_lb = cr_end_pt_info(:, 7);
    f_con_2der_ub = cr_end_pt_info(:, 8);
    
    % When l_vec_grp is within the convex region
    obf_fun_part = obj_fun_convex_1d(l_vec_grp, l_vec_old, d_vec, sc);
    
    % When l_vec_grp is to the left of the convex region
    left_of_cr = l_vec_grp < l_lb;
    l_diff = l_vec_grp(left_of_cr) - l_lb(left_of_cr);
    obj_fun_left = f_con_lb(left_of_cr) + f_con_der_lb(left_of_cr) .* l_diff + 0.5 * f_con_2der_lb(left_of_cr) .* (l_diff.^2);
    obf_fun_part(left_of_cr) = obj_fun_left;
    
    % When l_vec_grp is to the right of the convex region
    right_of_cr = l_vec_grp > l_ub;
    l_diff = l_vec_grp(right_of_cr) - l_ub(right_of_cr);
    obj_fun_right = f_con_ub(right_of_cr) + f_con_der_ub(right_of_cr) .* l_diff + 0.5 * f_con_2der_ub(right_of_cr) .* (l_diff.^2);
    obf_fun_part(right_of_cr) = obj_fun_right;    
    
    obf_fun_part = rho_loc .* obf_fun_part;
    
    if isempty(bias_term)
        %obf_fun_part(good_exp_loc)
        obj_fun = sum(obf_fun_part) + 0.5 * lambda * sum(theta .^ 2);
    else
        theta_no_bias = theta;
        theta_no_bias(bias_term) = 0;
        
        obj_fun = sum(obf_fun_part) + 0.5 * lambda * sum(theta_no_bias .^ 2);
    end
end

% Gradient function (convexified, split parameters)
function grad = grad_fun_param_split(theta, Xy, theta_hat, sc, d_vec, lambda, cr_end_pt_info, bias_term, l_vec_old, inv_rho_loc)
% Assumes theta, theta_hat, d_vec, l_vec_old, and inv_rho_loc are column vectors 
    l_vec = Xy * theta;
    l_vec_hat = Xy * theta_hat;
    
    %l_vec_grp = ((l_vec - l_vec_hat) .* inv_rho_loc) + l_vec_old;
    
    l_vec_grp = ((l_vec - l_vec_hat) .* inv_rho_loc);
    l_vec_grp(isnan(l_vec_grp)) = 0;
    l_vec_grp = l_vec_grp + l_vec_old;
    
    % Location of endpoints of convex region
    l_lb = cr_end_pt_info(:, 1);
    l_ub = cr_end_pt_info(:, 2);    
    f_con_der_lb = cr_end_pt_info(:, 5);
    f_con_der_ub = cr_end_pt_info(:, 6);
    f_con_2der_lb = cr_end_pt_info(:, 7);
    f_con_2der_ub = cr_end_pt_info(:, 8);
    
    % When l_vec_grp is within the convex region
    fder1_fun_l_vec = grad_fun_convex_1d(l_vec_grp, l_vec_old, d_vec, sc);
    
    % When l_vec_grp is to the left of the convex region
    left_of_cr = l_vec_grp < l_lb;
    l_diff = l_vec_grp(left_of_cr) - l_lb(left_of_cr);
    grad_part_left = f_con_der_lb(left_of_cr) + f_con_2der_lb(left_of_cr) .* l_diff;
    fder1_fun_l_vec(left_of_cr) = grad_part_left;
    
    % When l_vec_grp is to the right of the convex region
    right_of_cr = l_vec_grp > l_ub;
    l_diff = l_vec_grp(right_of_cr) - l_ub(right_of_cr);
    grad_part_right = f_con_der_ub(right_of_cr) + f_con_2der_ub(right_of_cr) .* l_diff;
    fder1_fun_l_vec(right_of_cr) = grad_part_right;     
    
    fder1_fun_l_vec(inv_rho_loc == 0) = 0;
    
    if isempty(bias_term)
        grad = fder1_fun_l_vec' * Xy + lambda * theta';
    else
        theta_no_bias = theta;
        theta_no_bias(bias_term) = 0;
        
        grad = fder1_fun_l_vec' * Xy + lambda * theta_no_bias';
    end
    grad = grad';
end

% Hessian times vector function (convexified)
function hess_vec = hessian_vector_mult_fun_param_split(theta, v, Xy, theta_hat, sc, d_vec, lambda, cr_end_pt_info, bias_term, l_vec_old, inv_rho_loc)
% Assumes theta, theta_hat, d_vec, l_vec_old, inv_rho_loc, and v are column vectors 
    l_vec = Xy * theta;
    l_vec_hat = Xy * theta_hat;
    
    %l_vec_grp = ((l_vec - l_vec_hat) .* inv_rho_loc) + l_vec_old;
    
    l_vec_grp = ((l_vec - l_vec_hat) .* inv_rho_loc);
    l_vec_grp(isnan(l_vec_grp)) = 0;
    l_vec_grp = l_vec_grp + l_vec_old;
    
    % Location of endpoints of convex region
    l_lb = cr_end_pt_info(:, 1);
    l_ub = cr_end_pt_info(:, 2);    
    f_con_2der_lb = cr_end_pt_info(:, 7);
    f_con_2der_ub = cr_end_pt_info(:, 8);
    
    % When l_vec_grp is within the convex region
    fder2_fun_l_vec = hess_fun_convex_1d(l_vec_grp, d_vec, sc);
    
    % When l_vec_grp is to the left of the convex region
    left_of_cr = l_vec_grp < l_lb;
    fder2_fun_l_vec(left_of_cr) = f_con_2der_lb(left_of_cr);
    
    % When l_vec_grp is to the right of the convex region
    right_of_cr = l_vec_grp > l_ub;
    fder2_fun_l_vec(right_of_cr) = f_con_2der_ub(right_of_cr);
    
    fder2_fun_l_vec = inv_rho_loc .* fder2_fun_l_vec;
        
    if isempty(bias_term)
        hess_vec = (fder2_fun_l_vec .* (Xy * v))' * Xy + lambda * v';
    else
        v_rem_bias = v;
        v_rem_bias(bias_term) = 0;
        
        hess_vec = (fder2_fun_l_vec .* (Xy * v))' * Xy + lambda * v_rem_bias';
    end
    hess_vec = hess_vec';
end

% Objective function (original)
function obj_fun = obj_fun_orig(theta, Xy, sc, lambda, bias_term)
% Assumes theta is a column vector
    l_vec = Xy * theta;
    
    if isempty(bias_term)
        obj_fun = sum(obj_fun_1d(l_vec, sc)) + ...
            0.5 * lambda * sum(theta .^ 2);
    else
        theta_no_bias = theta;
        theta_no_bias(bias_term) = 0;
        
        obj_fun = sum(obj_fun_1d(l_vec, sc)) + ...
            0.5 * lambda * sum(theta_no_bias .^ 2);        
    end
end

% Gradient function (original)
function grad = grad_fun_orig(theta, Xy, sc, lambda, bias_term)
% Assumes theta is a column vector 
    l_vec = Xy * theta;
    
    fder1_fun_l_vec = grad_fun_1d(l_vec, sc);
    
    if isempty(bias_term)
        grad = fder1_fun_l_vec' * Xy + lambda * theta';
    else
        theta_no_bias = theta;
        theta_no_bias(bias_term) = 0;
        
        grad = fder1_fun_l_vec' * Xy + lambda * theta_no_bias';
    end
    grad = grad';
end

% Objective function (1D convexified)
function obj_fun = obj_fun_convex_1d(v, l_vec_old, d_vec, sc)
    obj_fun = obj_fun_1d(v, sc) - 0.5 * d_vec .* ((v - l_vec_old).^2);
end

% Gradient function (1D convexified)
function grad = grad_fun_convex_1d(v, l_vec_old, d_vec, sc)
    grad = grad_fun_1d(v, sc) - d_vec .* (v - l_vec_old);
end

% Hessian function (1D convexified)
function hess = hess_fun_convex_1d(v, d_vec, sc)
    hess = hess_fun_1d(v, sc) - d_vec;
end

% Objective function (1D)
function obj_fun = obj_fun_1d(v, sc)
    obj_fun = 1 - tanh(sc * v);
end

% Gradient function (1D)
function grad = grad_fun_1d(v, sc)
    grad = -sc * (sech(sc * v)).^2;
end

% Hessian  function (1D)
function hess = hess_fun_1d(v, sc)
    hess = 2 * sc^2 * (sech(sc * v)).^2 .* tanh(sc * v);
end

function alpha_k = convex_polyhedron_line_segment_intersect(x_k, s_k, A, b)
%
% Inputs
% x_k       The current trust region iterate
% 
% s_k       The approximate step direction
%
% A         The matrix of coefficients in the linear inequality constraints
%
% b         The upper bounds in the linear inequality constraints
%
% Output
% alpha_k   The optimal stepsize along the approximate Newton direction for 
%           which the iterate lies in the feasible region
%
% This function finds the intersection point between a line segment and a
% convex polyhedron defined by Ax <= b. It uses two projections

% No constraints
if isempty(A) || isempty(b)
    alpha_k = 1;
    return;
end

% Define the needed projections
proj_x_k = A * x_k;
proj_s_k = A * s_k;

% Define the residual vector
res_k = b - proj_x_k;

% Check if x_k + s_k falls within the feasible region 
if all((proj_s_k - res_k) <= 0)
    alpha_k = 1;
    return;
end

alpha_set = res_k ./ proj_s_k;

good_set = alpha_set >= 0 & alpha_set <= 1;

alpha_k = min(alpha_set(good_set));

if isempty(alpha_k)
    error('The projection onto convex polyhedron failed. Investigate!!!');
end

end
