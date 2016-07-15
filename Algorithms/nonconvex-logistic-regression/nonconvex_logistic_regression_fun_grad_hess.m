function [obj_fun, grad, hess] = nonconvex_logistic_regression_fun_grad_hess(theta, Xy, lambda, eps)
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

% If necessary, compute the hessian matrix
if nargout > 2
    fder2_fun_l_vec = hess_fun_1d(l_vec);
    
    v = (2*(eps - theta .^ 2)) ./ ((theta .^ 2 + eps).^2);
    v(end) = 0;
    
    hess = Xy' * bsxfun(@times, fder2_fun_l_vec, Xy) + diag(lambda * v);
end

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