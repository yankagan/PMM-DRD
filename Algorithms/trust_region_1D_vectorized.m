function x_opt = trust_region_1D_vectorized(x_init, f_fun, g_fun, h_fun, options)
%
% Inputs:
%
% x_init    The initial starting points, one for each variable. Potentially
%           A vector
%
% f_fun     The function for computing the value of the objective at a
%           specific set of variables. A function handle. The output of
%           applying f_fun is potentially a vector. It has two inputs, the
%           set of variables and the index of the variables
%
% g_fun     The function for computing the gradient of the objective at a
%           specific set of variables. A function handle. The output of
%           applying g_fun is potentially a vector. It has two inputs, the
%           set of variables and the index of the variables
%
%
% h_fun     The function for computing the Hessian of the objective at a
%           specific set of variables. A function handle. The output of
%           applying h_fun is potentially a vector. It has two inputs, the
%           set of variables and the index of the variables
%
% options   A struct that has fields for the trust region parameters.
%           options.eta_0 > 0 determines when to update the iterate.
%           0 < options.eta_1 < options.eta_2 < 1 determine when to
%           increase or decrease the trust region options.Delta_k.
%           0 < options.sigma_1 < options.sigma_2 < 1 < options.sigma_3 are
%           constants that govern the update of options.Delta_k.
%
%           The custom settings are:
%           options.eta_0 = 1e-4
%           options.eta_1 = 0.25
%           options.eta_2 = 0.75
%           options.sigma_1 = 0.25
%           options.sigma_2 = 0.5
%           options.sigma_3 = 4.0           
%           options.Delta_k = abs(g_fun(x_init)), potentially a vector
%           options.max_iter = 1000
%
%           The gradient threshold for terminating the trust region
%           algorithm
%           options.grad_tol = 1e-8
%
%           The stepsize threshold for stopping the trust region algorithm
%           options.ss_tol = 1e-16
%
%           The function change threshold for stopping the trust region
%           algorithm
%           options.fun_tol = 1e-10
%
%           The trust region radius threshold for stopping the trust region
%           algorithm
%           options.rad_tol = 1e-10
%
%           The custom settings for the trust region Newton's method follow 
%           those used by Lin, Weng, and Keerthi in their 2008 paper,  
%           "Trust Region Newton Method for Large-Scale Logistic 
%           Regression". These settings are the same as those used by Lin  
%           and More in their 1999 paper "Newton's Method for Large 
%           Bound-Constrained Optimization Problems". The only difference  
%           is Lin and More uses options.eta_0 = 1e-3.              
%
% Output:
% x_opt     An optimal minimum point, one for each variable. Potentially
%           a vector
%
% This function implements a model trust region modified Newton's method in
% 1 dimension, using the approach proposed by C.-J. Lin and J.J. More in  
% their 1999 paper titled "Newton's Method for Large Bound-Constrained  
% Optimization Problems"
%
%
% 06/01/15

if nargin < 4
    error('You must specify the first 4 input parameters');
end
   
%**************************************************************************
%      Set default options by adding missing fields to options array
%**************************************************************************
if ~exist('options', 'var') || isempty(options)
    options = struct;
end

% The gradient threshold for terminating the trust region algorithm
if ~isfield(options, 'grad_tol') || isempty(options.grad_tol)    
    options.grad_tol = 1e-8;
end

% The stepsize threshold for stopping the trust region algorithm
if ~isfield(options, 'ss_tol') || isempty(options.ss_tol)    
    options.ss_tol = 1e-16;
end

% The function change threshold for terminating the trust region algorithm
if ~isfield(options, 'fun_tol') || isempty(options.fun_tol)    
    options.fun_tol = 1e-10;
end

% The trust region radius threshold for terminating the trust region 
% algorithm
if ~isfield(options, 'rad_tol') || isempty(options.rad_tol)    
    options.rad_tol = 1e-10;
end

% Maximum number of trust region iterations
if ~isfield(options, 'max_iter') || isempty(options.max_iter)    
    options.max_iter = 1000; 
end

% Constant that determines when to update the iterate.
if ~isfield(options, 'eta_0') || isempty(options.eta_0)
    options.eta_0 = 1e-4; 
end

% Constants that determine when to increase or decrease the trust region
if ~isfield(options, 'eta_1') || isempty(options.eta_1)
    options.eta_1 = 0.25; 
end
if ~isfield(options, 'eta_2') || isempty(options.eta_2)
    options.eta_2 = 0.75; 
end

% Constants that govern the update of options.Delta_k
if ~isfield(options, 'sigma_1') || isempty(options.sigma_1)
    options.sigma_1 = 0.25; 
end
if ~isfield(options, 'sigma_2') || isempty(options.sigma_2)
    options.sigma_2 = 0.5; 
end
if ~isfield(options, 'sigma_3') || isempty(options.sigma_3)
    options.sigma_3 = 4.0; 
end

% Let q be the number of parameters
q = numel(x_init);
need_update_idx = (1: q)';

% Determine the initial trust region size(s), one for each variable
if ~isfield(options, 'Delta_k') || isempty(options.Delta_k)
    options.Delta_k = abs(g_fun(x_init, need_update_idx)); 
end


% Trust region iteration

x_k = x_init;
fun_val_k = f_fun(x_k, need_update_idx); % q vector 

Delta_k = options.Delta_k;

for k = 1: options.max_iter
    
    if isempty(need_update_idx)
        break;
    end
      
    x_k_subset = x_k(need_update_idx);
    
    Delta_k_subset = Delta_k(need_update_idx);
    
    fun_val_k_subset = fun_val_k(need_update_idx);
    
    g_val_subset = g_fun(x_k, need_update_idx); % u vector. u <= q
    
    h_val_subset = h_fun(x_k, need_update_idx); % u vector. u <= q
    
    % Find an approximate solution s_k to the 1D trust region sub-problem
    s_k_subset = newton_step_1d(Delta_k_subset, g_val_subset, h_val_subset);    
        
    % Compute the actual and predicted function value reduction
    old_fun_val_k_subset = fun_val_k_subset;
    s_k = zeros(size(x_k));
    s_k(need_update_idx) = s_k_subset;
    new_fun_val_k_subset = f_fun(x_k + s_k, need_update_idx);
    act_red_k_subset = new_fun_val_k_subset - fun_val_k_subset;
    pred_red_k_subset = g_val_subset .* s_k_subset + 0.5 * s_k_subset.^2 .* h_val_subset;  % should be negative!  
    
    ratio_k_subset = act_red_k_subset ./ pred_red_k_subset; % Potentially a vector
    
    % Update x_k  
    upd_idx = ratio_k_subset > options.eta_0;    
    x_k_subset(upd_idx) = x_k_subset(upd_idx) + s_k_subset(upd_idx);    
    fun_val_k_subset(upd_idx) = new_fun_val_k_subset(upd_idx);
    
    if sum(upd_idx) ~= 0
        x_k(need_update_idx(upd_idx)) = x_k_subset(upd_idx);
        fun_val_k(need_update_idx(upd_idx)) = fun_val_k_subset(upd_idx);
    end
       
    
    % Update Delta_k
    % Choose Delta_k as gamma_k * |s_k|, where gamma_k is the minimum of a
    % quadratic that interpolates the function phi(gamma) = f_fun(x_k + 
    % gamma * s_k). This quadratic satisfies phi(0) = f_fun(x_k), 
    % phi'(0) = g_fun(x_k) * s_k, and phi(1) = f_fun(x_k + s_k). 
    % If phi(gamma) does not have a minimum, set gamma_k = +Inf.
    % Choose Delta_k as gamma_k * |s_k|, if it falls in the desired
    % interval [Delta_k_lb, Delta_k_ub]. Otherwise, set Delta_k to the 
    % closest endpoint to gamma_k.
    
    abs_s_k_subset = abs(s_k_subset);
    delta_case1 = ratio_k_subset <= options.eta_1;
    delta_case2 = (ratio_k_subset > options.eta_1) & (ratio_k_subset < options.eta_2);
    delta_case3 = ratio_k_subset >= options.eta_2;
    
    Delta_k_lb = zeros(size(abs_s_k_subset));
    Delta_k_ub = Delta_k_lb;
    
    Delta_k_lb(delta_case1) = options.sigma_1 * min([abs_s_k_subset(delta_case1), Delta_k_subset(delta_case1)], [], 2);
    Delta_k_ub(delta_case1) = options.sigma_2 * Delta_k_subset(delta_case1);
    
    Delta_k_lb(delta_case2) = options.sigma_1 * Delta_k_subset(delta_case2);
    Delta_k_ub(delta_case2) = options.sigma_3 * Delta_k_subset(delta_case2);
    
    Delta_k_lb(delta_case3) = Delta_k_subset(delta_case3);
    Delta_k_ub(delta_case3) = options.sigma_3 * Delta_k_subset(delta_case3);       
    
    
    % Find the minimum point of phi(gamma) = f_fun(x_k + gamma * s_k)
    q_b = g_val_subset .* s_k_subset;
    q_c = old_fun_val_k_subset;
    q_a = new_fun_val_k_subset - q_b - q_c;
    
    pos_q_a = q_a > 0;
    gamma_k_subset = Inf(size(q_a));    
    gamma_k_subset(pos_q_a) = -q_b(pos_q_a)./(2 * q_a(pos_q_a));
       
    Delta_k_subset = min([max([gamma_k_subset .* abs_s_k_subset, Delta_k_lb], [], 2), Delta_k_ub], [], 2); 
    Delta_k(need_update_idx) = Delta_k_subset;
    
    % If the actual function reduction is small, stop trying to update the
    % variable
    f_tol_need_update_idx = abs(act_red_k_subset) >= options.fun_tol;
    
    % If the gradient value is small, stop trying to update the variable
    g_tol_need_update_idx = abs(g_val_subset) >= options.grad_tol;
            
    % If the trust region size is too small, stop trying to update the
    % variable
    rad_need_update_idx = abs(Delta_k_subset) >= options.rad_tol;
        
    % Update the list of variables that need to be updated
    need_update_idx = need_update_idx(g_tol_need_update_idx & f_tol_need_update_idx & rad_need_update_idx);
end

x_opt = x_k;


end


%**************************************************************************
%                           Helper functions
%**************************************************************************
function s_k = newton_step_1d(Delta_k, g_val, h_val)
%
% Inputs
% Delta_k   The current trust region radius. Potentially a column vector
%
% g_val     The gradient at the current iterate. Potentially a column vector
%
% h_val     The Hessian at the current iterate. Potentially a column vector 
%
% Output
% s_k       The approximate Newton direction. Potentially a column vector
%
% This function determines the appropriate Newton or trust region truncated 
% Newton step.

s_k = -g_val ./ h_val;
s_k(isnan(s_k)) = 0; % 0/0 is defined as 0

% Make sure Newton step satisfies the trust region
s_k = min([s_k, Delta_k], [], 2);
s_k = max([s_k, -Delta_k], [], 2);

end
