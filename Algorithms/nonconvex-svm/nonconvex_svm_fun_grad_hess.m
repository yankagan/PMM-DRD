function [obj_fun, grad, hess] = nonconvex_svm_fun_grad_hess(theta, Xy, sc, lambda)
% Assumes theta is a column vector
l_vec = Xy * theta; 

obj_fun = sum(obj_fun_1d(l_vec, sc)) + ...
    0.5 * lambda * sum(theta(1: end - 1) .^ 2);
  
if nargout > 1
    fder1_fun_l_vec = grad_fun_1d(l_vec, sc);

    grad = fder1_fun_l_vec' * Xy + lambda * ([theta(1: end - 1); 0])';
    
    grad = grad';
end

% If necessary, compute the hessian matrix
if nargout > 2
    fder2_fun_l_vec = hess_fun_1d(l_vec, sc);
    
    p = numel(theta);
    p_minus_1 = p - 1;
    I = [eye(p_minus_1), zeros(p_minus_1, 1); zeros(1, p)];
    
    hess = Xy' * bsxfun(@times, fder2_fun_l_vec, Xy) + lambda * I;
end

end

%**************************************************************************
% Helper functions
%**************************************************************************

% Objective function (1D)
function obj_fun = obj_fun_1d(v, sc) %#ok<*DEFNU>
    obj_fun = 1 - tanh(sc * v);
end

% Gradient function (1D)
function grad = grad_fun_1d(v, sc)
    grad = -sc * (sech(sc * v)).^2;
end

% Hessian function (1D)
function hess = hess_fun_1d(v, sc)
    hess = 2 * sc^2 * (sech(sc * v)).^2 .* tanh(sc * v);
end