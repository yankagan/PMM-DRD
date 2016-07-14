%%% This code is a demo of the parallel majorization-minimization (PMM) algorithm for non-convex optimization published in    				      %%%
%%% Y. Kaganovsky, I. Odinaka, D. Carlson, and L. Carin, "Parallel Majorization Minimization with Dynamically Restricted Domains for Nonconvex Optimization", %%% 
%%% Journal of Machine Learning Research W&CP Vol. 51: Proc 19th conference on Artificial Intelligence and Statistics (AISTATS) 2016 		              %%%

%% Written by Ikenna Odinaka and Yan Kaganovsky 2016 %%

rng('default'); 

disp('Now Running Logistic Regression (LIBLINEAR for quadratic penalty and the rest for log penalty)');

 
% To be used for initialization 
Xy_train = bsxfun(@times,X_train,y_train); 
mag = max(sum(abs(Xy_train),1));

%parameter in log-penalty
eps = 0.1;

% For PMM
num_cores = 64;  
c_f = 0.1;
i_f = 0.8;
num_grp = ceil(p/500);
param_ord = 1: p;

% For PSCA
n_proc = 16;
n_blks_per_proc = 1; 

% Optimization options for all algorithms
options.outer_max_iter = 1000;
options.tol_fun = 1e-6;
options.tol_grad = 1e-8;
options.tol_step = 1e-10;
options.display = 0;
options.n_display = 1;
options.compute_obj = 1;
options.max_train_time = 60; %secs (1 minute)

% For trust region methods
options.max_iter = 10; % Trust region iteration
options.cg_max_iter = 10;
options.cg_xi_k = 1e-4;

% For C/C++ mex methods
options.num_outer_threads = min(num_cores, num_grp);
options.num_inner_threads = floor((num_cores - options.num_outer_threads)/options.num_outer_threads);

% For AdaGrad
options.adagrad_learning_rate = 1e-2;
options.adagrad_h_init = 0;

% For RMSProp
options.rmsprop_learning_rate = 1e-2;
options.rmsprop_h_init = 1e-2;

% For mini-batch methods
options.mini_batch_size = n;
options.learning_rate_exp = 1;

% regularization parameter
lambda = 1e-6;
  

fprintf('------------------------------------------------- \n');

%%% Repeat for different random initializations %%%

for ind_init = 1: no_inits
%**************************************************************************
%               Train all algorithms using the best lambda
%**************************************************************************
theta_0 = randn(p, 1)./mag;
fprintf('Running initialization %d/%d \n',ind_init,no_inits);
fprintf('------------------------------------------------- \n');

%%%%% PMM with Local curvature region  %%%%
options.display = 0;
% Trust Region
disp('Starting PMM ...');
[theta_opt_pmm_cg, obj_hist_pmm_cg, time_elapsed_pmm_cg] = nonconvex_logistic_regression_PMM_CG(X_train, y_train, theta_0, lambda, eps, num_grp, param_ord, c_f, i_f, options);
acc_test_pmm_cg = linear_classification_accuracy(X_test, y_test, theta_opt_pmm_cg);
acc_train_pmm_cg = linear_classification_accuracy(X_train(:,1:end-1), y_train, theta_opt_pmm_cg);
theta_opt_pmm_cg_CELL{ind_init}=theta_opt_pmm_cg; 
obj_hist_pmm_cg_CELL{ind_init}=obj_hist_pmm_cg;
time_elapsed_pmm_cg_CELL{ind_init}=time_elapsed_pmm_cg;
acc_train_pmm_cg_CELL{ind_init}=acc_train_pmm_cg;
acc_test_pmm_cg_CELL{ind_init}=acc_test_pmm_cg;
fprintf('Training/Test Accuracy for PMM was %f/%f (percent) \n',acc_train_pmm_cg, acc_test_pmm_cg);
fprintf('------------------------------------------------- \n');

options.n_display = 10;

%%%%% L-BFGS %%%%
disp('Starting LBFGS ...');
[theta_opt_lbfgs, obj_hist_lbfgs, time_elapsed_lbfgs] = nonconvex_logistic_regression_LBFGS(X_train, y_train, theta_0, lambda, eps, options);
acc_test_lbfgs = linear_classification_accuracy(X_test, y_test, theta_opt_lbfgs);
acc_train_lbfgs = linear_classification_accuracy(X_train(:,1:end-1), y_train, theta_opt_lbfgs);
theta_opt_lbfgs_CELL{ind_init}=theta_opt_lbfgs;
obj_hist_lbfgs_CELL{ind_init}=obj_hist_lbfgs;
time_elapsed_lbfgs_CELL{ind_init}=time_elapsed_lbfgs;
acc_train_lbfgs_CELL{ind_init}=acc_train_lbfgs;
acc_test_lbfgs_CELL{ind_init}=acc_test_lbfgs;
fprintf('Training/Test Accuracy for LBFGS was %f/%f (percent) \n',acc_train_lbfgs, acc_test_lbfgs);
fprintf('------------------------------------------------- \n');

%%%% Non-linear CG %%%%
disp('Starting CG ...');
[theta_opt_cg, obj_hist_cg, time_elapsed_cg] = nonconvex_logistic_regression_CG(X_train, y_train, theta_0, lambda, eps, options);
acc_test_cg = linear_classification_accuracy(X_test, y_test, theta_opt_cg);
acc_train_cg = linear_classification_accuracy(X_train(:,1:end-1), y_train, theta_opt_cg);
theta_opt_cg_CELL{ind_init}=theta_opt_cg;
obj_hist_cg_CELL{ind_init}=obj_hist_cg;
time_elapsed_cg_CELL{ind_init}=time_elapsed_cg;
acc_train_cg_CELL{ind_init}=acc_train_cg;
acc_test_cg_CELL{ind_init}=acc_test_cg;
fprintf('Training/Test Accuracy for CG was %f/%f (percent) \n',acc_train_cg, acc_test_cg);
fprintf('------------------------------------------------- \n');

%%%%%% GD %%%%%
disp('Starting GD ..');
[theta_opt_gd, obj_hist_gd, time_elapsed_gd] = nonconvex_logistic_regression_GD(X_train, y_train, theta_0, lambda, eps, options);
acc_test_gd = linear_classification_accuracy(X_test, y_test, theta_opt_gd);
acc_train_gd = linear_classification_accuracy(X_train(:,1:end-1), y_train, theta_opt_gd);
theta_opt_gd_CELL{ind_init}=theta_opt_gd;
obj_hist_gd_CELL{ind_init}=obj_hist_gd;
time_elapsed_gd_CELL{ind_init}=time_elapsed_gd;
acc_train_gd_CELL{ind_init}=acc_train_gd;
acc_test_gd_CELL{ind_init}=acc_test_gd;
fprintf('Training/Test Accuracy for GD was %f/%f (percent) \n',acc_train_gd, acc_test_gd);
fprintf('------------------------------------------------- \n');

%%%%%% AdaGrad %%%%%% 
disp('Starting Adagrad ...');
[theta_opt_adagrad, obj_hist_adagrad, time_elapsed_adagrad] = nonconvex_logistic_regression_mSGD_ADAgrad(X_train, y_train, theta_0, lambda, eps, options);
acc_test_adagrad = linear_classification_accuracy(X_test, y_test, theta_opt_adagrad);
acc_train_adagrad = linear_classification_accuracy(X_train(:,1:end-1), y_train, theta_opt_adagrad);
theta_opt_adagrad_CELL{ind_init}=theta_opt_adagrad;
obj_hist_adagrad_CELL{ind_init}=obj_hist_adagrad;
time_elapsed_adagrad_CELL{ind_init}=time_elapsed_adagrad;
acc_train_adagrad_CELL{ind_init}=acc_train_adagrad;
acc_test_adagrad_CELL{ind_init}=acc_test_adagrad;
fprintf('Training/Test Accuracy for RMSProp was %f/%f (percent) \n',acc_train_adagrad, acc_test_adagrad);
fprintf('------------------------------------------------- \n');

%%%%%% RMSProp %%%%%
disp('Starting RMSProp ...');
[theta_opt_rmsprop, obj_hist_rmsprop, time_elapsed_rmsprop] = nonconvex_logistic_regression_mSGD_RMSprop(X_train, y_train, theta_0, lambda, eps, options);
acc_test_rmsprop = linear_classification_accuracy(X_test, y_test, theta_opt_rmsprop);
acc_train_rmsprop = linear_classification_accuracy(X_train(:,1:end-1), y_train, theta_opt_rmsprop);
theta_opt_rmsprop_CELL{ind_init}=theta_opt_rmsprop;
obj_hist_rmsprop_CELL{ind_init}=obj_hist_rmsprop;
time_elapsed_rmsprop_CELL{ind_init}=time_elapsed_rmsprop;
acc_train_rmsprop_CELL{ind_init}=acc_train_rmsprop;
acc_test_rmsprop_CELL{ind_init}=acc_test_rmsprop;
fprintf('Training/Test Accuracy for RMSProp was %f/%f (percent) \n',acc_train_rmsprop, acc_test_rmsprop);
fprintf('------------------------------------------------- \n');

%%%%% PSCA %%%%
options.learning_rate_exp = 1;
disp('Starting PSCA ...');
[theta_opt_psca, obj_hist_psca, time_elapsed_psca] = nonconvex_logistic_regression_PSCA(X_train, y_train, theta_0, lambda, eps, n_proc, n_blks_per_proc, options);
acc_test_psca = linear_classification_accuracy(X_test, y_test, theta_opt_psca);
acc_train_psca = linear_classification_accuracy(X_train(:,1:end-1), y_train, theta_opt_psca);
theta_opt_psca_CELL{ind_init}=theta_opt_psca;
obj_hist_psca_CELL{ind_init}=obj_hist_psca;
time_elapsed_psca_CELL{ind_init}=time_elapsed_psca;
acc_train_psca_CELL{ind_init}=acc_train_psca;
acc_test_psca_CELL{ind_init}=acc_test_psca;
fprintf('Training/Test Accuracy for PSCA was %f/%f (percent) \n',acc_train_psca, acc_test_psca);
fprintf('------------------------------------------------- \n');


%%%%%% SAVE RESULTS %%%%%%%%%%
save results_TB_nonconvex_LR ...
    acc_train_pmm_cg_CELL ...
    acc_test_pmm_cg_CELL ...
    ...
    acc_train_lbfgs_CELL ...
    acc_test_lbfgs_CELL ...
    ...
    acc_train_cg_CELL ...
    acc_test_cg_CELL ...
    ...
    acc_train_psca_CELL ...
    acc_test_psca_CELL ...
    ...
    acc_train_gd_CELL ...
    acc_test_gd_CELL ...
    ...    
    acc_train_adagrad_CELL ... 
    acc_test_adagrad_CELL ...
    ...
    acc_train_rmsprop_CELL ...
    acc_test_rmsprop_CELL 
end



%%%%%%%%%%%%%%%%  Done with Algorithms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



