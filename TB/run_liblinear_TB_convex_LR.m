%% Written by Ikenna Odinaka and Yan Kaganovsky 2016 %%

% This script performs cross-validation using the dataset to find
% the best set of regularization parameter based on a grid search for
% LIBLINEAR


fprintf('Running Liblinear for Logistic Regression \n');

tol_opt = 1e-4;

lam_best = 0.2;

% Train 
options_liblin = ['-q -s 6 -c ', num2str(lam_best) '-e ' num2str(tol_opt)];
t = tic;
model = train(y_train, X_train, options_liblin);
train_time = toc(t);

% Classification accuracy on training data
fprintf('Training ');
[pred_label, acc_vec, ~] = predict(y_train, X_train, model);
acc_train = acc_vec(1);


% Classification accuracy on test data
fprintf('Test ');
[pred_label, acc_vec, ~] = predict(y_test, X_test, model);
acc_test = acc_vec(1);


save results_TB_convex_LR acc_train acc_test
