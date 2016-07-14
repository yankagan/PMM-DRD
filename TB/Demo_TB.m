%%% This code is a demo of the parallel majorization-minimization (PMM) algorithm for non-convex optimization published in    				      %%%
%%% Y. Kaganovsky, I. Odinaka, D. Carlson, and L. Carin, "Parallel Majorization Minimization with Dynamically Restricted Domains for Nonconvex Optimization", %%% 
%%% Journal of Machine Learning Research W&CP Vol. 51: Proc 19th conference on Artificial Intelligence and Statistics (AISTATS) 2016 		              %%%

%% Written by Ikenna Odinaka and Yan Kaganovsky 2016 %%

clear all
close all
clc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% compilex mex files and take care of paths %%%%%%%%%%%%%%%%%%%%%%%%%%%
% compile LIBLIN package
run('../Algorithms/liblinear-1.96/matlab/make.m');

% compile MinFunc package for convex optimization by Mark Schmidt
run('../Algorithms/Mark-Schmidt-Code/mexAll.m'); 

% compile our mex files for the proposed algorithm
run('../Algorithms/mex-functions/nonconvex-svm-pmm-cg/compile_mex.m');
run('../Algorithms/mex-functions/nonconvex-logistic-regression-pmm-cg/compile_mex.m');


% Path to algorithms
addpath(genpath('../Algorithms')); %


%%%%%%%%%%%%%%%%%%%%%%%%% Load Dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% assemble parts of dataset into one %
load TB_dataset_hiv_class_X_test_part1.mat X_test1;
load TB_dataset_hiv_class_X_test_part2.mat X_test2;
load TB_dataset_hiv_class_X_val_part1.mat X_val1;
load TB_dataset_hiv_class_X_val_part2.mat X_val2;
load TB_dataset_hiv_class_X_train_part1.mat X_train1;
load TB_dataset_hiv_class_X_train_part2.mat X_train2;
load TB_dataset_hiv_class_X_train_part3.mat X_train3;
load TB_dataset_hiv_class_X_train_part4.mat X_train4;
load TB_dataset_hiv_class_X_train_part5.mat X_train5;
load TB_dataset_hiv_class_y y_train y_val y_test;

X_train=[X_train1 X_train2 X_train3 X_train4 X_train5];
X_test=[X_test1 X_test2];
X_val=[X_val1 X_val2];

% merge validation and training into one big training set
X_train = [X_train; X_val]; 
y_train = [y_train; y_val];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% run algorithms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%% Support Vector Machines %%%%%%%%%%%%%%%%%%%%

% number of random initializations for each algorithm in SVM
no_inits=1;    % Originally was 10; Default is 1 for speed
fprintf('Running %d different initializations (Default is 1 for speed) \n',no_inits);
fprintf('------------------------------------------------- \n');

% Run the script that calls all algorithms for non-convex SVM
run('run_all_algorithms_TB_nonconvex_SVM.m');


% Run the script that calls LIBLIN for L2 penalized hinge-loss (convex) SVM
run('run_liblinear_TB_convex_SVM.m');

% Display results in a table
fprintf('------------------------------------------------------------------- \n');
fprintf('Summary of Results for SVM and TB Dataset                           \n');
fprintf('------------------------------------------------------------------- \n');
filename='results_TB_convex_svm';
display_results_in_table(filename);
fprintf('Note: LIBLINEAR is for the convex SVM and the rest are for the nonconvex sigmoid-loss SVM \n');

fprintf('Press any key to continue to logistic regression \n');
pause
clc

%%%%%%%%%%%% Logistic Regressions %%%%%%%%%%%%%%%%%%%%

% number of random initializations for each algorithm in LR
no_inits=1;    % Originally was 10; Default is 1 for speed
fprintf('Running %d different initializations (Default is 1 for speed) \n',no_inits);
fprintf('------------------------------------------------- \n');

% Run the script that calls all algorithms for non_convex logistic regression
run('run_all_algorithms_TB_nonconvex_LR.m');

% Run the script that calls LIBLIN for L1 penalized logistic regression
run('run_liblinear_TB_convex_LR.m');

% Display results in a table
fprintf('------------------------------------------------------------------ \n');
fprintf('Summary of Results for Logistic Regression and TB Dataset          \n');
fprintf('------------------------------------------------------------------ \n');
filename='results_TB_nonconvex_LR';
display_results_in_table(filename);
fprintf('Note: LIBLINEAR is for the quadratic penalty and the rest are for the nonconvex log penalty \n');
