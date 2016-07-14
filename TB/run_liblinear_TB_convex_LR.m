%% Written by Ikenna Odinaka and Yan Kaganovsky 2016 %%

% This script performs cross-validation using the dataset to find
% the best set of regularization parameter based on a grid search for
% LIBLINEAR

% Cross validation parameters
log10_l_min = -13; 
log10_l_max = 13; 
num_l = log10_l_max - log10_l_min + 1;


fprintf('Runing Liblinear for Logistic Regression \n');

load TB_dataset_hiv_class.mat y_train X_train y_val X_val y_test X_test

log10_lam_range = linspace(log10_l_min, log10_l_max, num_l);   

n_param = numel(log10_lam_range);

cv_acc = NaN(n_param, 1);

tol_opt = 1e-4;

lam_best = 0.2;

% Train 
options_liblin = ['-q -s 6 -c ', num2str(lam_best) '-e ' num2str(tol_opt)];
t = tic;
model = train([y_train; y_val], [X_train; X_val], options_liblin);
train_time = toc(t);

% Classify test data
[pred_label, acc_vec, ~] = predict(y_test, X_test, model);
acc = acc_vec(1);

save results_TB_convex_LR lam_best model pred_label acc train_time
