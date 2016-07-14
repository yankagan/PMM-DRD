function x_opt = trust_region_ND_CG_unbounded(x_init, f_fun, g_fun, hv_fun, options)
%
% Inputs:
%
% x_init    The initial starting point. An N-vector
%
% f_fun     The function for computing the value of the objective at a
%           specific point. A function handle
%
% g_fun     The function for computing the gradient of the objective at a
%           specific point. A function handle
%
% hv_fun    The function for computing the product of the Hessian of the 
%           objective at a specific point and a vector. A function handle.
%           The first input to hv_fun is the specific point at which the 
%           Hessian is to be computed, while the second input is the vector 
%           which the Hessian matrix is to multiply.
%
% options   A struct that has fields for the trust region parameters.
%           options.eta_0 > 0 determines when to update the iterate.
%           0 < options.eta_1 < options.eta_2 < 1 determine when to
%           increase or decrease the trust region options.Delta_k.
%           0 < options.sigma_1 < options.sigma_2 < 1 < options.sigma_3 are
%           constants that govern the update of options.Delta_k.
%
%           The custom settings (for trust region Newton's method) are:
%           options.eta_0 = 1e-4
%           options.eta_1 = 0.25
%           options.eta_2 = 0.75
%           options.sigma_1 = 0.25
%           options.sigma_2 = 0.5
%           options.sigma_3 = 4.0           
%           options.Delta_k = norm(g_fun(x_init))
%           options.max_iter = 10
%
%           The gradient threshold for terminating the trust region
%           algorithm
%           options.grad_tol = 1e-8
%
%           The function change threshold for stopping the trust region
%           algorithm
%           options.fun_tol = 1e-10
%
%           The trust region radius threshold for stopping the trust region
%           algorithm
%           options.rad_tol = 1e-10
%
%           The custom settings (for the conjugate gradient procedure) are:
%           options.cg_xi_k = 1e-4 
%           options.cg_max_iter = 10
%
%           Whether to display messages
%           options.display = 1
%
%           The custom settings for the trust region Newton's method and 
%           the conjugate gradient procedure follow those used by Lin, 
%           Weng, and Keerthi in their 2008 paper, "Trust Region Newton 
%           Method for Large-Scale Logistic Regression". These settings are 
%           the same as those used by Lin and More in their 1999 paper  
%           "Newton's Method for Large Bound-Constrained Optimization
%           Problems". The only difference is Lin and More uses 
%           options.eta_0 = 1e-3.
%
% Output:
% x_opt     An optimal minimum point. An N x 1 vector
%
% This function implements a model trust region modified Newton's method in
% N dimensions, using the approach proposed by C.-J. Lin and J.J. More in  
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
    options.max_iter = 10; 
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

% Determine the initial trust region size
if ~isfield(options, 'Delta_k') || isempty(options.Delta_k)
    options.Delta_k = norm(g_fun(x_init)); 
end

% Conjugate gradient (inner iterations) parameters
if ~isfield(options, 'cg_xi_k') || isempty(options.cg_xi_k)
    options.cg_xi_k = 1e-4;
end
if ~isfield(options, 'cg_max_iter') || isempty(options.cg_max_iter)
    options.cg_max_iter = 10; 
end

% Whether to display messages
if ~isfield(options, 'display') || isempty(options.display)
    options.display = 1;
end

% Trust region iteration
x_k = x_init;
fun_val_k = f_fun(x_k);

for k = 1: options.max_iter
    g_vec = g_fun(x_k);
    
    if norm(g_vec) < options.grad_tol
        if options.display
            fprintf('TR: Optimality condition for gradient norm, grad_tol, reached\n');
        end
        
        break;
    end     
    
    % Find an approximate solution s_k to the ND trust region sub-problem
    s_k = conjugate_gradient(x_k, options.Delta_k, options.cg_xi_k, ...
        options.cg_max_iter, g_vec, hv_fun, options.display);
    
        
    % Compute the actual and predicted function values
    new_fun_val_k = f_fun(x_k + s_k);
    act_red_k = new_fun_val_k - fun_val_k;
    pred_red_k = g_vec' * s_k + 0.5 * s_k' * hv_fun(x_k, s_k);  % should be negative!  
    
    ratio_k = act_red_k/pred_red_k;
    
    % Update x_k    
    old_fun_val_k = fun_val_k;
    if ratio_k > options.eta_0
        x_k = x_k + s_k;
        fun_val_k = new_fun_val_k;
    end
    
    % Update options.Delta_k
    % Choose Delta_k as gamma_k * ||s_k||, where gamma_k is the minimum of 
    % a quadratic that interpolates the function phi(gamma) = f_fun(x_k + 
    % gamma * s_k). This quadratic satisfies phi(0) = f_fun(x_k), 
    % phi'(0) = g_fun(x_k)' * s_k, and phi(1) = f_fun(x_k + s_k). 
    % If phi(gamma) does not have a minimum, set gamma_k = +Inf.
    % Choose Delta_k as gamma_k * ||s_k||, if it falls in the desired
    % interval [Delta_k_lb, Delta_k_ub]. Otherwise, set Delta_k to the 
    % closest endpoint to gamma_k.
    
    norm_s_k = norm(s_k);
    if ratio_k <= options.eta_1
        Delta_k_lb = options.sigma_1 * min(norm_s_k, options.Delta_k);
        Delta_k_ub = options.sigma_2 * options.Delta_k;
    elseif ratio_k < options.eta_2
        Delta_k_lb = options.sigma_1 * options.Delta_k;
        Delta_k_ub = options.sigma_3 * options.Delta_k;
    else
        Delta_k_lb = options.Delta_k;
        Delta_k_ub = options.sigma_3 * options.Delta_k;       
    end
    
    % Find the minimum point of phi(gamma) = f_fun(x_k + gamma * s_k)
    q_b = g_vec' * s_k;
    q_c = old_fun_val_k;
    q_a = new_fun_val_k - q_b - q_c;
    
    if q_a > 0
        gamma_k = -q_b/(2 * q_a);
    else
        gamma_k = Inf;
    end     
    options.Delta_k = min(max(gamma_k * norm_s_k, Delta_k_lb), Delta_k_ub);  
    
    % If the actual function reduction is small, exit
    if abs(act_red_k) < options.fun_tol
        if options.display
            fprintf('TR: Optimality condition for function value, fun_tol, reached\n');
        end
        
        break;
    end
    
    % If the trust region size is too small, exit
    if abs(options.Delta_k) < options.rad_tol
        if options.display
            fprintf('TR: Optimality condition for trust region radius, rad_tol, reached\n');
        end
        
        break;
    end
end

x_opt = x_k;

end


%**************************************************************************
%                           Helper functions
%**************************************************************************
function s_k = conjugate_gradient(x_k, Delta_k, xi_k, max_iter, g_vec, hv_fun, display)
%
% Inputs
% x_k       The current trust region iterate
%
% Delta_k   The current trust region radius
%
% xi_k      A factor of the gradient norm used in stopping the cg procedure
%
% max_iter  The maximum number of cg iterations
%
% g_vec     The gradient vector at the current iterate
%
% hv_fun    The Hessian times vector function handle 
%
% display   Whether to display messages
%
% Output
% s_k       The approximate Newton direction
%
% This function performs the inner conjugate gradient procedure for solving
% the linear equation obtained from solving for the Newton direction.

n = numel(x_k);
s_i = zeros(n, 1);
r_i = -g_vec;
d_i = r_i;

grad_tol = xi_k * norm(g_vec);

for  i = 1: max_iter
    
    norm_r_i = norm(r_i);
    
    if norm_r_i <= grad_tol
        s_k = s_i;
        
        if display
            fprintf('CG: Optimality condition for gradient value, grad_tol, reached\n');
        end
        
        return;
    end
    
    hess_times_d_i = hv_fun(x_k, d_i);
    
    norm_r_i_sq = norm_r_i * norm_r_i;
    
    alpha_i = norm_r_i_sq / (d_i' * hess_times_d_i);
    
    s_i_plus_1 = s_i + alpha_i * d_i;
    
    if norm(s_i_plus_1) >= Delta_k
        
        % Find tau such that ||s_i + tau * d_i|| = Delta_k
        a = d_i' * d_i;
        b = 2*(s_i' * d_i);
        c = s_i' * s_i - (Delta_k * Delta_k);
        
        % Find the biggest root of the quadratic
        tau = max(roots([a, b, c]));
        
        s_k = s_i + tau * d_i;
        
        if display
            fprintf('CG: Optimality condition for trust region radius\n');
        end
        
        return;
    end    
    s_i = s_i_plus_1;
    
    r_i = r_i - alpha_i * hess_times_d_i;
    
    beta_i = (norm(r_i) .^ 2) / norm_r_i_sq;
    
    d_i = r_i + beta_i * d_i;    
end
s_k = s_i;

end
