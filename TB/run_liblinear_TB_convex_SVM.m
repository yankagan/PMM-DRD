%% Written by Ikenna Odinaka and Yan Kaganovsky 2016 %%

% This script performs cross-validation using the dataset to find
% the best set of regularization parameter based on a grid search for
% LIBLINEAR

% Cross validation parameters
log10_l_min = -13; 
log10_l_max = 13; 
num_l = log10_l_max - log10_l_min + 1;

fprintf('Running Liblinear for SVM \n');

load TB_dataset_hiv_class.mat y_train X_train y_val X_val y_test X_test

tol_opt = 1e-4;

lam_best = 0.001; 

% Train 
options_liblin = ['-q -s 1 -c ', num2str(lam_best) '-e ' num2str(tol_opt)];
t = tic;
model = train([y_train; y_val], [X_train; X_val], options_liblin);
train_time = toc(t);

% Classification accuracy on training data
fprintf('Training ');
[pred_label, acc_vec, ~] = predict(y_train, X_train, model);
acc_train = acc_vec(1);


% Classification accuracy on test data
fprintf('Test ');
[pred_label, acc_vec, ~] = predict(y_test, X_test, model);
acc_test = acc_vec(1);



save results_TB_convex_svm acc_test acc_train 

