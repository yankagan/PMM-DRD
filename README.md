# Non-Convex Optimization for Large-Scale Embedded Feature Selection in Binary Classification  

We provide a demo of the parallel majorization-minimization (PMM) algorithm for non-convex optimization in machine learning that was published in [1]. This paper presents an optimization framework for nonconvex problems based on majorization-minimization that is particularity well-suited for cases when the number of features/parameters is very large. The original large-scale problem is broken into smaller sub-problems that can be solved in parallel while still guaranteeing the monotonic reduction of the original objective function and convergence to a local minimum. Due to the low dimensionality of each sub-problem, second-order optimization methods become computationally feasible and can be used to accelerate convergence.

The advantages of this algorithm over other gradient-based methods for non-convex optimization (e.g., Gradient Descent, LBFGS, RMS-Prop, ADAGRAD) are as follows:
(1) No tuning of learning rates required;
(2) Scale invariance (similar to 2nd order methods but without using the Hessian);
(3) Highly parallel - the parameters are split into small groups that can be processed in parallel;
(4) Cost function guarenteed to decrease monotonically (any increase is an indication of a bug) without any expensive line searches.

Although the framework in [1] address a wide range of cost functions in the generalized linear model family, the current code only supports two specific models: sigmoid-loss SVM and logistic regression with logarithmic penalty (for promoting stronger sparsity) applied to binary classification problems. We hope to make the code more general in the future.


References
-----------
1. Y. Kaganovsky, I. Odinaka, D. Carlson, and L. Carin, Parallel Majorization Minimization with Dynamically Restricted Domains for Nonconvex Optimization, Journal of Machine Learning Research W&CP Vol. 51: Proc 19th conference on Artificial Intelligence and Statistics (AISTATS) 2016.

How to Run the Code
--------------------
Open MATLAB and run Demo_TB.m from the TB folder. It will compile the C++ mex files during the first run.
Tested on Linux and Windows machines running MATLAB 8.4. 

Comments
---------
1. The code uses openMP in the C++ code to perform parallel computations in order to obtain a considerable speedup. This feature will be turned off when using compilers that do not support openMP, resulting in a considerable slow-down of the PMM algorithm.
2. Currently all algorithms for the non-convex models use a single random initialization. To run multiple random initializations and compute the mean and variance for the classification accuracy, the user can modify the variable "no_inits" in the file Demo_TB.m.
3. In the paper [1], the run time for each algorithm was limited to 5 minutes. To save time, we shortened it to 1 minute in logistic regression so some discreptincies realtive to [1] are expected. To obtain the same results, the user can change the parameter options.max_train_time = 300 (secs) in the file run_all_algorithms_TB_nonconvex_LR.m. 
