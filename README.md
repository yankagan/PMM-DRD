# Non-Convex Optimization for Large-Scale Embedded Feature Selection in Machine learning 

We provide a demo of the parallel majorization-minimization (PMM) algorithm for non-convex optimization in machine learning that was published in [1] .This paper presents an optimization framework for nonconvex problems based on majorization-minimization that is particularity well-suited for cases when the number of features/parameters is very large. The original large-scale problem is broken into smaller sub-problems that can be solved in parallel while still guaranteeing the monotonic reduction of the original objective function and convergence to a local minimum. Due to the low dimensionality of each sub-problem, second-order optimization methods become computationally feasible and can be used to accelerate convergence.

The advantages of this algorithm over other gradient-based methods for non-convex optimization (e.g., Gradient Descent, LBFGS, RMS-Prop, ADAGRAD) are as follows:
(1) No tuning of learning rates required;
(2) Scale invariance (similar to 2nd order methods but without using the Hessian);
(3) Highly parallel - the parameters are split into small groups that can be processed in parallel;
(4) Cost function guarenteed to decrease monotonically (any increase is an indication of a bug) without any expensive line searches.

Although the framework in [1] address a wide range of cost functions in the generalized linear model family, the current code only supports two specific models: sigmoid-loss SVM and logistic regression with logarithmic penalty (for promoting stronger sparsity). We hope to make the code more general in the future.


References
-----------
1. Y. Kaganovsky, I. Odinaka, D. Carlson, and L. Carin, Parallel Majorization Minimization with Dynamically Restricted Domains for Nonconvex Optimization, Journal of Machine Learning Research W&CP Vol. 51: Proc 19th conference on Artificial Intelligence and Statistics (AISTATS) 2016.
