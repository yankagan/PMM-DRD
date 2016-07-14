function [intervals, f2der_min_interval] = ...
    get_lower_upper_proj_bounds(proj_vec, gmp_vec, f2der_min, f2der_fun, ...
    c_f, i_f, inflex_pts, pos_curv_thr)
%
%
% Inputs:
% proj_vec      The projection of the current parameter estimate. The 
%               argument to f(). The i-th element of proj_vec is l_i   
%
% gmp_vec       The locations of the argument to f(l) which achieve the
%               minimum negative curvature f2der_min
% 
% f2der_min     The minimum negative curvature
%
% f2der_fun     The handle to the 1D second derivative function f''(l)
%
% c_f           The curvature factor. 0 <= c_f < 1. This factor will be
%               used to multiply f2der_min to set up a threshold above
%               which l_i's with negative second derivative will have a
%               designed local convexity region (LCR). This LCR has been 
%               defined to deal with slow update issues that arise at 
%               points with negative and large curvature (close to 0) 
%               relative to the minimum curvature point. 
%               NB: If c_f is 0, no point will satisfy the wide concave
%               region requirement, (so no speed up for such regions). If
%               c_f is 1, all projections with negative curvature will
%               satisfy this requirement, so no escape from this region is
%               possible. So c_f is not allowed to be 1
%
% i_f           The interval factor.  0 < i_f <= 1. This factor defines how
%               far from the minimum curvature point(s) we want the lower
%               and upper bounds defining the local CR to be
%
% inflex_pts    The inflexion points of f(l). That is, where f''(l) = 0
%
% pos_curv_thr  The threshold in f'' below which a projection in a convex
%               region (region with f'' > 0) is allowed to escape the 
%               confines of the consecutive inflexion points of f(l)
%
% Outputs:
%
% intervals     A matrix with two columns, defining the lower and upper
%               bounds, respectively, for each l_i
%
% f2der_min_interval The minimum curvature in each interval
%
% 06/03/15

% This is subtracted from f2der_min_interval to account for round-off 
% error in computing the minimum negative curvative in the convex region.
epsilon = 1e-6;

if ~exist('c_f', 'var') || isempty(c_f)
    c_f = 0.5;
else
    if c_f >= 1 || c_f < 0 
        warning('c_f must lie in the interval [0, 1). Setting c_f to 0 (No speed-up for wide regions)');
        c_f = 0;
    end
end

if ~exist('i_f', 'var') || isempty(i_f)
    i_f = 0.5;
else
    if i_f > 1 || i_f <= 0
        warning('i_f must lie in the interval (0, 1]. Setting i_f = 1 (No speed-up for wide regions)');
        i_f = 1;
    end
end

if ~exist('inflex_pts', 'var')
    inflex_pts = [];
end

if ~exist('pos_curv_thr', 'var') || isempty(pos_curv_thr)
    pos_curv_thr = max(c_f, 0.1) * abs(f2der_min);
else
   if pos_curv_thr <= 0
       warning('pos_curv_thr must be positive to avoid getting trapped between consecutive inflexion points. Setting to default value');
       pos_curv_thr = max(c_f, 0.1) * abs(f2der_min);
   end
end

Nl = numel(proj_vec);

% Default interval
intervals = [-Inf(Nl, 1), Inf(Nl, 1)];

% Default 2nd derivative for all negative curvature regions
f2der_min_interval = f2der_min * ones(Nl, 1);

if isempty(gmp_vec)
    % Use global convex region
    f2der_min_interval = f2der_min_interval - epsilon * abs(f2der_min);
    
    %fprintf('Using global convex region');
    return;
end

% Find f''(l)
f2der_l = f2der_fun(proj_vec);

% Prepend -Inf and append +Inf to gmp_vec
gmp_vec = gmp_vec(:);
gmp_vec = [-Inf; gmp_vec; Inf];

%**************************************************************************
%   Projection with negative but large curvature (close to zero). 
%                    Concave and Wide Regions (CvWR)
%**************************************************************************
% Find where f''(l) < 0 and f''(l) > c_f * f2der_min
cvwr = (f2der_l < 0) & (f2der_l > c_f * f2der_min);

% Find a smaller CR around the problem l_i's
proj_vec_subset = proj_vec(cvwr);
proj_vec_subset = proj_vec_subset(:);

[lcr, min_curv_lcr] = local_convex_region(gmp_vec, proj_vec_subset, i_f, f2der_fun);

intervals(cvwr, :) = lcr;
f2der_min_interval(cvwr) = min_curv_lcr;

%**************************************************************************
%   Projections with positive curvature. 
%
%   Case 1. If curvature above a certain level, set the convex region to
%   only include regions with positive curvature between consecutive 
%   inflexion points. So the minimum curvature in the local convex region 
%   (LCR) will be 0. So called, convex and narrow region (CxNR)
%
%   Case 2. If curvature is close enough to zero, allow the LCR to include 
%   points of negative curvature to allow for possible escape from the 
%   positive curvature region. Use the same update rule as in the concave
%   and wide region above. So called, convex and wide region (CxWR)
%
%   Case 3. When the curvature is very high (higher than abs(f2der_min). Use
%   the global convex region
%**************************************************************************
if ~isempty(inflex_pts)
    % Prepend -Inf and append +Inf to inflex_pts
    inflex_pts = inflex_pts(:);
    inflex_pts = [-Inf; inflex_pts; Inf];
    
    % Case 1
    % Find where f''(l) > 0 and f''(l) >= pos_curv_thr and f''(l) <=
    % abs(f2der_min)
    cxnr = (f2der_l > 0) & (f2der_l >= pos_curv_thr) & (f2der_l <= abs(f2der_min));    
    
    proj_vec_subset = proj_vec(cxnr);
    proj_vec_subset = proj_vec_subset(:);
    
    % Use i_f = 1 here so that the projections are not trapped in case 1
    [lcr, min_curv_lcr] = local_convex_region(inflex_pts, proj_vec_subset, 1, f2der_fun);
    
    intervals(cxnr, :) = lcr;
    f2der_min_interval(cxnr) = min_curv_lcr;
    
    % Case 2
    % Find where f''(l) > 0 and f''(l) < pos_curv_thr
    cxwr = (f2der_l > 0) & (f2der_l < pos_curv_thr); 
    
    proj_vec_subset = proj_vec(cxwr);
    proj_vec_subset = proj_vec_subset(:);
    
    [lcr, min_curv_lcr] = local_convex_region(gmp_vec, proj_vec_subset, i_f, f2der_fun);

    intervals(cxwr, :) = lcr;
    f2der_min_interval(cxwr) = min_curv_lcr;   
    
    % Case 3
    % Find where f''(l) > 0 and f''(l) > abs(f2der_min)
    cxgcr = (f2der_l > 0) & (f2der_l > abs(f2der_min));
    
    intervals(cxgcr, 1) = -Inf;
    intervals(cxgcr, 2) = Inf;
    f2der_min_interval(cxgcr) = f2der_min; 
end


% To ensure positive definiteness in the convexified objective
f2der_min_interval = f2der_min_interval - epsilon * abs(f2der_min);

end

%**************************************************************************
% Helper function
%**************************************************************************
% Obtain local convex region for problematic regions
function [lcr, min_curv_lcr] = local_convex_region(interest_pt, proj_vec, i_f, f2der_fun)
% Assumes interest_point and proj_vec are column vectors
    l_min_is_gt_l = bsxfun(@gt, interest_pt, proj_vec');
    l_min_is_le_l = ~l_min_is_gt_l;
    l_ub_logical = l_min_is_gt_l & [zeros(1, size(l_min_is_le_l, 2)); l_min_is_le_l(1: end - 1, :)];
    a_idx_rep = repmat((1: numel(interest_pt))', [1, numel(proj_vec)]);
    l_ub_idx = a_idx_rep(l_ub_logical); % Index of upper bound to the l_i's
    l_lb_idx = l_ub_idx - 1; % Index of lower bound to the l_i's

    smaller_interval_lb = i_f * interest_pt(l_lb_idx) + (1 - i_f) * proj_vec;
    smaller_interval_ub = i_f * interest_pt(l_ub_idx) + (1 - i_f) * proj_vec;

    lcr = [smaller_interval_lb, smaller_interval_ub];

    % Get the smallest curvature in each interval
    lcr_finite = ~isinf(lcr);
    f2der_interval = zeros(size(lcr));
    f2der_interval(lcr_finite) = f2der_fun(lcr(lcr_finite));
    
    min_curv_lcr = min(f2der_interval, [], 2);
end
