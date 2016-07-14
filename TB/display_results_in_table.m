%% Written Yan Kaganovsky 2016 %%

function display_results_in_table(file)

fprintf('------------------------------------------------------------------- \n');
fprintf('Summary of Results for SVM and TB Dataset                           \n');
fprintf('------------------------------------------------------------------- \n');


load(file);
acc_train_liblin=acc_train;
acc_test_liblin=acc_test;

filename{1}='results_TB_nonconvex_SVM';

no_inits=1;
fprintf('Classification accuracy (in percent) using the format: MEAN (STD) for %d initializations \n',no_inits);
fprintf('Method || Train Accuracy || Test Accuracy \n',no_inits);

load(filename{1});

acc_test_pmm_cg_mat=cell2mat(acc_test_pmm_cg_CELL); 
acc_test_pmm_cg_mean=mean(acc_test_pmm_cg_mat,2);
acc_test_pmm_cg_std=std(acc_test_pmm_cg_mat,[],2);
acc_train_pmm_cg_mat=cell2mat(acc_train_pmm_cg_CELL); 
acc_train_pmm_cg_mean=mean(acc_train_pmm_cg_mat,2);
acc_train_pmm_cg_std=std(acc_train_pmm_cg_mat,[],2);


acc_test_lbfgs_mat=cell2mat(acc_test_lbfgs_CELL);
acc_test_lbfgs_mean=mean(acc_test_lbfgs_mat,2);
acc_test_lbfgs_std=std(acc_test_lbfgs_mat,[],2);    
acc_train_lbfgs_mat=cell2mat(acc_train_lbfgs_CELL);
acc_train_lbfgs_mean=mean(acc_train_lbfgs_mat,2);
acc_train_lbfgs_std=std(acc_train_lbfgs_mat,[],2);    

acc_test_cg_mat=cell2mat(acc_test_cg_CELL);
acc_test_cg_mean=mean(acc_test_cg_mat,2);
acc_test_cg_std=std(acc_test_cg_mat,[],2);
acc_train_cg_mat=cell2mat(acc_train_cg_CELL);
acc_train_cg_mean=mean(acc_train_cg_mat,2);
acc_train_cg_std=std(acc_train_cg_mat,[],2);

acc_test_psca_mat=cell2mat(acc_test_psca_CELL);
acc_test_psca_mean=mean(acc_test_psca_mat,2);
acc_test_psca_std=std(acc_test_psca_mat,[],2);
acc_train_psca_mat=cell2mat(acc_train_psca_CELL);
acc_train_psca_mean=mean(acc_train_psca_mat,2);
acc_train_psca_std=std(acc_train_psca_mat,[],2);

acc_test_gd_mat=cell2mat(acc_test_gd_CELL);
acc_test_gd_mean=mean(acc_test_gd_mat,2);
acc_test_gd_std=std(acc_test_gd_mat,[],2);
acc_train_gd_mat=cell2mat(acc_train_gd_CELL);
acc_train_gd_mean=mean(acc_train_gd_mat,2);
acc_train_gd_std=std(acc_train_gd_mat,[],2);
    
acc_test_adagrad_mat=cell2mat(acc_test_adagrad_CELL);
acc_test_adagrad_mean=mean(acc_test_adagrad_mat,2);
acc_test_adagrad_std=std(acc_test_adagrad_mat,[],2);    
acc_train_adagrad_mat=cell2mat(acc_train_adagrad_CELL);
acc_train_adagrad_mean=mean(acc_train_adagrad_mat,2);
acc_train_adagrad_std=std(acc_train_adagrad_mat,[],2);    

acc_test_rmsprop_mat=cell2mat(acc_test_rmsprop_CELL);
acc_test_rmsprop_mean=mean(acc_test_rmsprop_mat,2);
acc_test_rmsprop_std=std(acc_test_rmsprop_mat,[],2);
acc_train_rmsprop_mat=cell2mat(acc_train_rmsprop_CELL);
acc_train_rmsprop_mean=mean(acc_train_rmsprop_mat,2);
acc_train_rmsprop_std=std(acc_train_rmsprop_mat,[],2);

fprintf(' LIBLINEAR ||  %f  || %f \n L-BFGS ||  %f (%f) || %f (%f) \n CG ||  %f (%f) || %f (%f) \n GD ||  %f (%f) || %f (%f)\n PSCA || %f (%f) || %f (%f) \n RMSProp || %f (%f) || %f (%f) \n AdaGrad || %f (%f) || %f (%f)\n PMM || %f (%f) || %f (%f)\n', ...
acc_train_liblin, acc_test_liblin, acc_train_lbfgs_mean, acc_train_lbfgs_std, acc_test_lbfgs_mean, acc_test_lbfgs_std, ...
acc_train_cg_mean, acc_train_cg_std, acc_test_cg_mean, acc_test_cg_std, ...
acc_train_gd_mean, acc_train_gd_std, acc_test_gd_mean, acc_test_gd_std, ...
acc_train_psca_mean, acc_train_psca_std, acc_test_psca_mean, acc_test_psca_std, ... 
acc_train_rmsprop_mean, acc_train_rmsprop_std, acc_test_rmsprop_mean, acc_test_rmsprop_std, ...
acc_train_adagrad_mean, acc_train_adagrad_std, acc_test_adagrad_mean, acc_test_adagrad_std, ...
acc_train_pmm_cg_mean, acc_train_pmm_cg_std, acc_test_pmm_cg_mean, acc_test_pmm_cg_std);

fprintf('------------------------------------------------- \n');
fprintf('Note: LIBLINEAR is for the convex SVM and the rest are for the nonconvex sigmoid-loss SVM \n');
end
