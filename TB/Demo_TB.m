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
run('../Algorithms/Mark Schmidt Code/mexAll.m'); 

% compile our mex files for the proposed algorithm
run('../Algorithms/mex functions/nonconvex svm pmm cg/compile_mex.m');
run('../Algorithms/mex functions/nonconvex logistic regression pmm cg/compile_mex.m');


% Path to algorithms
addpath(genpath('../Algorithms')); %

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
display_results_in_table(fileneame);
