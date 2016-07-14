// This code is a demo of the parallel majorization-minimization (PMM) algorithm for non-convex optimization published in    				      //
// Y. Kaganovsky, I. Odinaka, D. Carlson, and L. Carin, "Parallel Majorization Minimization with Dynamically Restricted Domains for Nonconvex Optimization",  // 
// Journal of Machine Learning Research W&CP Vol. 51: Proc 19th conference on Artificial Intelligence and Statistics (AISTATS) 2016 		              //

// Written by Yan Kaganovsky and Ikenna Odinaka 2016 //

/*
PMM_CG_nonconvex_logistic_regression.cpp  

Usage:
	theta_out = PMM_CG_nonconvex_logistic_regression(Xy_grp, theta, grp_splits, cr_end_pt_info, l_vec, d_vec, lambda, inv_rho_grp, bias_grp, bias_idx, eps, options, num_outer_threads, num_inner_threads)

Inputs: 
	Xy_grp			A cell array of matrices. It has num_grp cells. Each cell corresponds to a parameter group. Each matrix has I rows. 
					The sum of the number of columns in each  matrix across the cells is J

	theta			The previous parameter vector estimate. The expansion point. It has J elements
	
	grp_splits		Splitting point used in determining the parameters in each group. It has (num_grp + 1) elements

	cr_end_pt_info	A matrix with 8 columns containing the lower endpoint, upper endpoint, function value at both endpoints, first derivative at both endpoints, 
					and second derivative at both endpoints of the convexified 1D functions. The end points define a local convex region in the parameter space
					It has dimensions: J x 8 (stored columnwise)

	l_vec			The previous projections. l_vec = Xy * theta. It has I elements
	
	d_vec			The negative of the minimum curvature in each parameter. It has J elements
	
	lambda			The regularization coefficient
	
	inv_rho_grp		The reciprocal of the convexification coefficients used in splitting the paramaters over groups. 
					Its a matrix of dimensions: I x num_grp (stored columnwise)
	
	bias_grp		The index of the group containing the bias term. A scalar

	bias_idx		The index within the group of the bias term. A scalar
	
	eps				The shift parameter of the nonconvex log penalty

	options			Options to be use in the Trust Region (TR) Newton's Method with Conjugate Gradient for determining the search directions. 
					A struct that has fields:
					options.eta_0 > 0  -> Determines when to update the iterate.
					0 < options.eta_1 < options.eta_2 < 1  -> Determine when to increase or decrease the trust region radius.
					0 < options.sigma_1 < options.sigma_2 < 1 < options.sigma_3 -> Constants that govern the update of the trust region radius.
					options.max_iter -> The maximum number of TR iterations
					options.grad_tol -> The gradient threshold for terminating the trust region algorithm
					options.fun_tol -> The function change threshold for stopping the trust region algorithm
					options.rad_tol -> The trust region radius threshold for stopping the trust region algorithm
          
					For the conjugate gradient (CG) procedure:
					options.cg_xi_k -> A parameter for terminating CG
					options.cg_max_iter -> The maximum number of CG iterations
          
					options.display -> Whether to display messages

					The Trust region Newton's method and the conjugate gradient procedure follow those used by Lin, 
					Weng, and Keerthi in their 2008 paper, "Trust Region Newton Method for Large-Scale Logistic Regression". 

num_inner_threads	The number of threads used in parallelizing over groups of parameters, available for OpenMP to use. A scalar

num_outer_threads	The number of threads used within each individual parameter group, available for OpenMP to use. A scalar


Output:
	theta_out		The current parameter vector estimate.

Modified: 11/19/15 

Authors: 
  
This cpp routine implements the inside parallelized loop (to avoid the computational and memory overhead in MATLAB's parfor) 
for parallel majorization minimization (PMM) using a Trust Region Newton's Method with Conjugate Gradient for determining the search directions. 

*/

// For size_t, include stddef.h, which is included by stdio.h
// For mwSize, include matrix.h
// For mex functions, you must include mex.h
// compile with: /openmp
#include <iostream>
#include <algorithm> /*For min and max*/
#include <limits>	/* For float epsilon and inf */
#include <cstdlib>  /*For system pause */
#include <stdio.h>
#include <stdlib.h> /* calloc, malloc, free*/
// #include <math.h>
#include <cmath> /* fmin and fmax are not supported by Visual C++ 2012 and some 2013 */
#include "matrix.h"
#include "mex.h"


#define ENABLE_OMP defined(_OPENMP)

#if ENABLE_OMP
	// OpenMP headers
	#include <omp.h>
#endif

// Define variable checking MACROs
#define IS_REAL_2D_FULL_DOUBLE(X) ( !mxIsComplex(X) && mxGetNumberOfDimensions(X) == 2 && !mxIsSparse(X) && mxIsDouble(X) )
#define IS_REAL_1D_FULL_DOUBLE(X) ( IS_REAL_2D_FULL_DOUBLE(X) && (mxGetM(X) == 1 || mxGetN(X) == 1) )
#define IS_REAL_SCALAR_DOUBLE(X)  ( IS_REAL_2D_FULL_DOUBLE(X) && mxGetNumberOfElements(X) == 1 )
#define IS_CELL_ARRAY_OF_REAL_MATRIX(X) (mxGetClassID(X) == mxCELL_CLASS && mxGetNumberOfElements(X) > 0 && !mxIsComplex(mxGetCell(X, 0)))
#define IS_STRUCT(X) (mxGetClassID(X) == mxSTRUCT_CLASS)

using namespace std;

extern void _main(); // main function is defined somewhere else.

// Computational routines:

// A C/C++ struct for optimization parameters
struct Options
{
	static int display;
	static int max_iter;
	static int cg_max_iter;
	static double eta_0;
	static double eta_1;
	static double eta_2;
	static double sigma_1;
	static double sigma_2;
	static double sigma_3; 
	static double fun_tol;
	static double grad_tol;
	static double rad_tol;
	static double cg_xi;
};

// Initialize the parameters
int Options::display = 1;
int Options::max_iter = 10;
int Options::cg_max_iter = 10;
double Options::eta_0 = 1e-4;
double Options::eta_1 = 0.25;
double Options::eta_2 = 0.75;
double Options::sigma_1 = 0.25;
double Options::sigma_2 = 0.5;
double Options::sigma_3 = 4.0; 
double Options::fun_tol = 1e-10;
double Options::grad_tol = 1e-8;
double Options::rad_tol = 1e-10;
double Options::cg_xi = 1e-4; 


/** A function for computing the L2 norm of a vector
	\arg v is the vector whose L2 norm is to be computed
	\arg N is the number of elements in v
	\returns the L2 norm
*/
static double l2_norm(const double * v, const int N)
{
	double sum = 0;
	for (int i = 0; i < N; i++)
	{
		const double v_i = v[i];
		sum += v_i * v_i;
	}
	return sqrt(sum);
}


/** A function for computing the L2 norm of a vector
	\arg u is the first vector
	\arg v is the second vector
	\arg N is the number of elements in v
	\returns the inner product
*/
static double inner_product(const double * u, const double * v, const int N)
{
	double sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum += u[i] * v[i];
	}
	return sum;
}


// Sparse matrix dense vector product routine
// The serial version of the code was adapted from the smvp.c MATLAB mex code written by Darren Engwirda Copyright 2006
// A is a matrix and b is a dense vector. c = A * b is the output vector
// This code assumes c is already initialized to zeros for the serial version and the parallel version based on atomics
static void sparse_matrix_dense_vector_product(double* c, const mxArray *A, const double* b, const int n_threads = 0)
{
	// Error checking
	const int N = mxGetN(A);

	if (!mxIsSparse(A))
		mexErrMsgIdAndTxt("MATLAB:sparse_matrix_dense_vector_product:sparsity_mismatch", "A must be sparse");

	const mwIndex *ir, *jc;
	const double *nz;

	ir = mxGetIr(A);      /* Row indexing      */
	jc = mxGetJc(A);      /* Column count      */
	nz = mxGetPr(A);      /* Non-zero elements */

	if (n_threads > 0)
	{
		// Parallel implementation(s)

		
		// 1) Using Atomic
		// Loop through columns
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int i = 0; i < N; i++)
		{
			const int start_idx = jc[i];
			const int stop_idx = jc[i + 1];
			const double b_i = b[i];

			//Loop through nonzero elements in i-th column
			for (int k = start_idx; k < stop_idx; k++)
			{
				#pragma omp atomic
				c[ir[k]] += nz[k] * b_i;
			}
		}		
		
		/*
		// 2) Using private copies of c
		// calloc may not be thread safe in threaded environments above the current set of threads
		const int M = mxGetM(A);
		// c_mat has dimensions: n_threads x M
		double *c_mat = (double*)calloc(M*n_threads, sizeof(double));
		if (c_mat == NULL)
			exit(1);

		// Loop through columns
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int i = 0; i < N; i++)
		{
			const int start_idx = jc[i];
			const int stop_idx = jc[i + 1];
			const double b_i = b[i];

			// Get the thread number of the thread executing within its thread team
			int t = omp_get_thread_num();

			// Get the t-th row of c_mat (row-major indexing)
			double *c_mat_t = &c_mat[M * t];

			//Loop through nonzero elements in i-th column
			for (int k = start_idx; k < stop_idx; k++)
			{
				c_mat_t[ir[k]] += nz[k] * b_i;
			}
		}

		// Combine the private copies of c
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int i = 0; i < M; i++)
		{
			double sum = 0.0;
			for (int t = 0; t < n_threads; t++)
			{
				sum += c_mat[M * t + i];
			}
			c[i] = sum;
		}
		*/		
	}
	else
	{
		// Serial implementation
		// Loop through columns
		for (int i = 0; i < N; i++)
		{
			const int start_idx = jc[i];
			const int stop_idx = jc[i + 1];
			const double b_i = b[i];

			//Loop through nonzero elements in i-th column
			for (int k = start_idx; k < stop_idx; k++)
			{
				c[ir[k]] += nz[k] * b_i;
			}
		}
	}
}


// Dense matrix dense vector product routine
// A is a dense matrix and b is a dense vector. c = A * b is the output vector
// This code assumes c is already initialized to zeros 
static void dense_matrix_dense_vector_product(double* c, const mxArray *A, const double* b, const int n_threads = 0)
{
	const int m = mxGetM(A);
	const int n = mxGetN(A);

	const double *A_p = mxGetPr(A);

	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int j = 0; j < n; j++) // run down the entries of b
		{
			const double b_j = b[j];

			// Get a pointer to the jth column of A
			const double *A_p_j = &A_p[j * m];

			for (int i = 0; i < m; i++)
			{
				#pragma omp atomic
				c[i] += A_p_j[i] * b_j;
			}
		}

		/*
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int i = 0; i < m; i++) // run down the rows of A
		{
			double sum = 0;
			for (int j = 0; j < n; j++)
			{
				sum += A_p[i + j * m] * b[j];
			}
			c[i] = sum;
		}
		*/
	}
	else
	{
		for (int j = 0; j <n; j++) // run down the entries of b
		{
			const double b_j = b[j];

			// Get a pointer to the jth column of A
			const double *A_p_j = &A_p[j * m];

			for (int i = 0; i <m; i++)
			{
				c[i] += A_p_j[i] * b_j;
			}
		}
	}
}


// Dense vector sparse matrix product routine
// A is a matrix and b is a dense vector. c = b * A is the output vector
// c does not have to be initialized
static void dense_vector_sparse_matrix_product(double* c, const mxArray *A, const double* b, const int n_threads = 0)
{
	// Error checking
	const int M = mxGetM(A);
	const int N = mxGetN(A);

	if (!mxIsSparse(A))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:dense_vector_sparse_matrix_product:sparsity_mismatch", "A must be sparse");

	const mwIndex *ir, *jc;
	const double *nz;

	ir = mxGetIr(A);      /* Row indexing */
	jc = mxGetJc(A);      /* Column count */
	nz = mxGetPr(A);      /* Non-zero elements */

	if (n_threads > 0)
	{
		// Loop over columns
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int i = 0; i < N; i++)
		{
			const int start_idx = jc[i];
			const int stop_idx = jc[i + 1];

			double sum = 0.0;
			// Loop through nonzero elements in i-th column
			for (int k = start_idx; k < stop_idx; k++)
			{
				sum += nz[k] * b[ir[k]];
			}

			c[i] = sum;
		}
	}
	else
	{
		// Serial implementation
		// Loop through columns
		for (int i = 0; i < N; i++)
		{
			const int start_idx = jc[i];
			const int stop_idx = jc[i + 1];

			double sum = 0.0;
			// Loop through nonzero elements in i-th column
			for (int k = start_idx; k < stop_idx; k++)
			{
				sum += nz[k] * b[ir[k]];
			}

			c[i] = sum;
		}
	}
}


/** Dense vector and dense matrix product routine
	\arg A is a dense matrix
	\arg b is a dense vector
	\arg n_threads is the number of threads to use in parallel processing
	\returns c = b * A, the output vector
	c does not have to be initialized, but should exist in memory
*/
static void dense_vector_dense_matrix_product(double* c, const mxArray *A, const double* b, const int n_threads = 0)
{
	// Error checking
	const mwSize M = mxGetM(A);
	const mwSize N = mxGetN(A);

	const double *A_p = mxGetPr(A);      /* elements */

	if (n_threads > 0)
	{
		// Loop over columns
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int i = 0; i < N; i++)
		{
			double sum = 0.0;

			// Get the i-th column of A
			const double *A_p_i = &A_p[i*M];

			// Loop through nonzero elements in i-th column
			for (int k = 0; k < M; k++)
			{
				//sum += A_p[i*M + k] * b[k];
				sum += A_p_i[k] * b[k];
			}
			c[i] = sum;
		}
	}
	else
	{
		// Serial implementation
		// Loop through columns
		for (int i = 0; i < N; i++)
		{
			double sum = 0.0;
			
			// Get the i-th column of A
			const double *A_p_i = &A_p[i*M];

			// Loop through nonzero elements in i-th column
			for (int k = 0; k < M; k++)
			{
				//sum += A_p[i*M + k] * b[k];
				sum += A_p_i[k] * b[k];
			}
			c[i] = sum;
		}
	}
}


// A namespace that contains functors for the 1D convexified (by adding a quadratic with curvature equal to negative the minimum curvature d) function, its derivative and 2nd derivative
// This only accounts for the log penalty part
namespace Convexified1D
{
	/** Functor for the convexified function
	*/
	template<typename type>
	class Function1D
	{
	private:
		type eps;
		type lambda;
	public:
		/** Constructor
			\arg eps_ is the shift parameter
			\arg lambda_ is the regularization constant
		*/
		Function1D(type eps_, type lambda_) : eps(eps_), lambda(lambda_)
		{
		}

		/** Evaluator
			\arg v is an input to the function
			\args l and d are function parameters
			\returns the function value
		*/
		type operator()(type v, type l, type d) const
		{
			
			return lambda * (log(v * v + eps) - 0.5 * d * pow(v - l, 2));
		}

		/** Setter
			\arg eps_ is the shift parameter
			\arg lambda_ is the regularization constant
		*/
		void set_variables(type eps_, type lambda_)
		{
			eps = eps_;
			lambda = lambda_;
		}
	};

	template<typename type>
	class Gradient1D
	{
	private:
		type eps;
		type lambda;
	public:
		/** Constructor
			\arg eps_ is the shift parameter
			\arg lambda_ is the regularization constant
		*/
		Gradient1D(type eps_, type lambda_) : eps(eps_), lambda(lambda_)
		{
		}

		/** Evaluator
			\arg v is an input to the function
			\args l and d are function parameters
			\returns the function value
		*/
		type operator()(type v, type l, type d) const
		{
			return lambda * ((2 * v) / (v * v + eps) - d * (v - l));
		}

		/** Setter
			\arg eps_ is the shift parameter
			\arg lambda_ is the regularization constant
		*/
		void set_variables(type eps_, type lambda_)
		{
			eps = eps_;
			lambda = lambda_;
		}
	};

	template<typename type>
	class Hessian1D
	{
	private:
		type eps;
		type lambda;
	public:
		/** Constructor
			\arg eps_ is the shift parameter
			\arg lambda_ is the regularization constant
		*/
		Hessian1D(type eps_, type lambda_) : eps(eps_), lambda(lambda_)
		{
		}

		/** Evaluator
			\arg v is an input to the function
			\arg d is a function parameter
			\returns the function value
		*/
		type operator()(type v, type d) const
		{
			return lambda * ((2 * (eps - v * v)) / (pow(v * v + eps,  2)) - d);
		}

		/** Setter
			\arg eps_ is the shift parameter
			\arg lambda_ is the regularization constant
		*/
		void set_variables(type eps_, type lambda_)
		{
			eps = eps_;
			lambda = lambda_;
		}
	};
}


/** A function for computing the objective function based on the quadratic extension
	\arg l_vec is an un-initialized vector that stores 1D projections of theta. It needs to be initialized to zeros in the code
	\arg l_vec_hat is an un-initialized vector that stores 1D projections of theta. It needs to be initialized to zeros in the code
	\arg theta is the current parameter values for the current group
	\arg Xy	is the feature matrix for the current group
	\arg theta_hat is the previous parameter values for the current group
	\arg eps is the shift parameter of the penalty function
	\arg d_vec is the minimum (local range) second derivative for each parameter value
	\arg lambda is the regularization coefficient
	\arg cr_end_pt_info is a JJ x 8 matrix with end-point information for each of the parameter values
	\arg bias_term specifies the location of the bias term (if any) within the current group. A negative value signals none
	\arg l_vec_old is the vector of previous 1D projections using all the parameters
	\arg inv_rho_loc is the reciprocal of the convexifying coefficient for the current group
	\arg JJ is the total number of parameters
	\arg start_idx is the starting index of the first parameter in the current group of parameters within the entire parameter vector
	\arg n_threads is the number of threads to use for parallel processing
	\returns the objective function value in obj_fun
*/
static void obj_fun_param_split(double &obj_fun, double *l_vec, double *l_vec_hat, const double *theta, const mxArray *Xy, const double *theta_hat, const double eps,
	const double *d_vec, const double lambda, const double *cr_end_pt_info, const int bias_term, const double *l_vec_old, const double *inv_rho_loc, 
	const int JJ, const int start_idx, const int n_threads = 0)
{
	// Initialize the objective function
	obj_fun = 0;

	// Initialize the functor for the 1D objective function
	Convexified1D::Function1D<double> func(eps, lambda);

	const size_t I = mxGetM(Xy);
	const size_t J = mxGetN(Xy);

	// Initialize the arrays to store the 1D projections of theta and theta_hat to zero
	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int i = 0; i < I; i++)
		{
			l_vec[i] = 0;
			l_vec_hat[i] = 0;
		}
	}
	else
	{
		for (int i = 0; i < I; i++)
		{
			l_vec[i] = 0;
			l_vec_hat[i] = 0;
		}
	}

	// Compute matrix vector products
	if (mxIsSparse(Xy))
	{
		sparse_matrix_dense_vector_product(l_vec, Xy, theta, n_threads);
		sparse_matrix_dense_vector_product(l_vec_hat, Xy, theta_hat, n_threads);
	}
	else
	{
		dense_matrix_dense_vector_product(l_vec, Xy, theta, n_threads);
		dense_matrix_dense_vector_product(l_vec_hat, Xy, theta_hat, n_threads);
	}

	// Get a pointer to the columns of cr_end_pt_info
	const double *t_lb = &cr_end_pt_info[0 + start_idx];
	const double *t_ub = &cr_end_pt_info[JJ + start_idx];
	const double *f_con_lb = &cr_end_pt_info[2 * JJ + start_idx];
	const double *f_con_ub = &cr_end_pt_info[3 * JJ + start_idx];
	const double *f_con_der_lb = &cr_end_pt_info[4 * JJ + start_idx];
	const double *f_con_der_ub = &cr_end_pt_info[5 * JJ + start_idx];
	const double *f_con_2der_lb = &cr_end_pt_info[6 * JJ + start_idx];
	const double *f_con_2der_ub = &cr_end_pt_info[7 * JJ + start_idx];

	// Get a pointer to the correct offset in d_vec
	const double *d_vec_offset = &d_vec[start_idx];

	// Compute data-fit part
	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int i = 0; i < I; i++)
		{
			double l_grp, rho = 0.0;

			const double l_old = l_vec_old[i];
			const double inv_rho = inv_rho_loc[i];

			if (inv_rho > 0)
			{
				l_grp = (l_vec[i] - l_vec_hat[i]) * inv_rho + l_old;
				rho = 1.0 / inv_rho;
			}
			else
				l_grp = l_old;

			// Incorporate the convexity factor
			if (rho > 0)
			{
				// Prevent exp(-x) from blowing up in log(1 + exp(-x))
				if (l_grp < -700)
				{
					#pragma omp atomic 
					obj_fun -= rho * l_grp;
				}
				else
				{
					#pragma omp atomic 
					obj_fun += rho * log(1 + exp(-l_grp));
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < I; i++)
		{
			double l_grp, rho = 0.0;

			const double l_old = l_vec_old[i];
			const double inv_rho = inv_rho_loc[i];

			if (inv_rho > 0)
			{
				l_grp = (l_vec[i] - l_vec_hat[i]) * inv_rho + l_old;
				rho = 1.0 / inv_rho;
			}
			else
				l_grp = l_old;

			// Incorporate the convexity factor
			if (rho > 0)
			{ 
				// Prevent exp(-x) from blowing up in log(1 + exp(-x))
				if (l_grp < -700)
				{
					obj_fun -= rho * l_grp;
				}
				else
				{
					obj_fun += rho * log(1 + exp(-l_grp));
				}
			}
		}
	}

	// Regularizer part
	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int j = 0; j < J; j++)
		{
			const double d_j = d_vec_offset[j];
			const double t_lb_j = t_lb[j];
			const double t_ub_j = t_ub[j];
			const double f_con_lb_j = f_con_lb[j];
			const double f_con_ub_j = f_con_ub[j];
			const double f_con_der_lb_j = f_con_der_lb[j];
			const double f_con_der_ub_j = f_con_der_ub[j];
			const double f_con_2der_lb_j = f_con_2der_lb[j];
			const double f_con_2der_ub_j = f_con_2der_ub[j];

			double obj_fun_reg, t_diff;

			double theta_j = theta[j];

			if (bias_term != j)
			{
				if (theta_j <= t_lb_j)
				{
					// Quadratic extension
					t_diff = theta_j - t_lb_j;

					obj_fun_reg = f_con_lb_j + f_con_der_lb_j * t_diff + 0.5 * f_con_2der_lb_j * (t_diff * t_diff);
				}
				else if (theta_j >= t_ub_j)
				{
					// Quadratic extension
					t_diff = theta_j - t_ub_j;

					obj_fun_reg = f_con_ub_j + f_con_der_ub_j * t_diff + 0.5 * f_con_2der_ub_j * (t_diff * t_diff);
				}
				else
				{
					// Convexified 1D objective
					double theta_hat_j = theta_hat[j];
					obj_fun_reg = func(theta_j, theta_hat_j, d_j);
				}

				#pragma omp atomic
				obj_fun += obj_fun_reg;
			}
		}
	}
	else
	{
		for (int j = 0; j < J; j++)
		{
			const double d_j = d_vec_offset[j];
			const double t_lb_j = t_lb[j];
			const double t_ub_j = t_ub[j];
			const double f_con_lb_j = f_con_lb[j];
			const double f_con_ub_j = f_con_ub[j];
			const double f_con_der_lb_j = f_con_der_lb[j];
			const double f_con_der_ub_j = f_con_der_ub[j];
			const double f_con_2der_lb_j = f_con_2der_lb[j];
			const double f_con_2der_ub_j = f_con_2der_ub[j];

			double obj_fun_reg, t_diff;

			double theta_j = theta[j];

			if (bias_term != j)
			{
				if (theta_j <= t_lb_j)
				{
					// Quadratic extension
					t_diff = theta_j - t_lb_j;

					obj_fun_reg = f_con_lb_j + f_con_der_lb_j * t_diff + 0.5 * f_con_2der_lb_j * (t_diff * t_diff);
				}
				else if (theta_j >= t_ub_j)
				{
					// Quadratic extension
					t_diff = theta_j - t_ub_j;

					obj_fun_reg = f_con_ub_j + f_con_der_ub_j * t_diff + 0.5 * f_con_2der_ub_j * (t_diff * t_diff);
				}
				else
				{
					// Convexified 1D objective
					double theta_hat_j = theta_hat[j];
					obj_fun_reg = func(theta_j, theta_hat_j, d_j);
				}

				obj_fun += obj_fun_reg;
			}
		}
	}
}


/** A function for computing the gradient of the objective function based on the quadratic extension
	\arg l_vec is an un-initialized vector that stores 1D projections of theta. It needs to be initialized to zeros in the code
	\arg l_vec_hat is an un-initialized vector that stores 1D projections of theta. It needs to be initialized to zeros in the code
	\arg grad_fun_part is an un-initialized vector that stores 1D gradients. It needs to be initialized to zeros in the code
	\arg theta is the current parameter values for the current group
	\arg Xy	is the feature matrix for the current group
	\arg theta_hat is the previous parameter values for the current group
	\arg eps is the shift parameter of the penalty function
	\arg d_vec is the minimum (local range) second derivative for each parameter value
	\arg lambda is the regularization coefficient
	\arg cr_end_pt_info is a JJ x 8 matrix with end-point information for each of the parameter values
	\arg bias_term specifies the location of the bias term (if any) within the current group. A negative value signals none
	\arg l_vec_old is the vector of previous 1D projections using all the parameters
	\arg inv_rho_loc is the reciprocal of the convexifying coefficient for the current group
	\arg JJ is the total number of parameters
	\arg start_idx is the starting index of the first parameter in the current group of parameters within the entire parameter vector
	\arg n_threads is the number of threads to use for parallel processing
	\returns the gradient of the objective function in grad
	This function assumes grad has already been created in memory
*/
static void grad_fun_param_split(double *grad, double *l_vec, double *l_vec_hat, double *grad_fun_part, const double *theta, const mxArray *Xy, const double *theta_hat, const double eps,
	const double *d_vec, const double lambda, const double *cr_end_pt_info, const int bias_term, const double *l_vec_old, const double *inv_rho_loc, 
	const int JJ, const int start_idx, const int n_threads = 0)
{
	// Initialize the functor for the 1D derivative function
	Convexified1D::Gradient1D<double> func(eps, lambda);

	const size_t I = mxGetM(Xy);
	const size_t J = mxGetN(Xy);

	// Initialize the arrays to store the 1D projections of theta and theta_hat and vector of 1D gradients to zero
	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int i = 0; i < I; i++)
		{
			l_vec[i] = 0;
			l_vec_hat[i] = 0;
			grad_fun_part[i] = 0;
		}
	}
	else
	{
		for (int i = 0; i < I; i++)
		{
			l_vec[i] = 0;
			l_vec_hat[i] = 0;
			grad_fun_part[i] = 0;
		}
	}
	
	// Compute matrix vector products
	if (mxIsSparse(Xy))
	{
		sparse_matrix_dense_vector_product(l_vec, Xy, theta, n_threads);
		sparse_matrix_dense_vector_product(l_vec_hat, Xy, theta_hat, n_threads);
	}
	else
	{
		dense_matrix_dense_vector_product(l_vec, Xy, theta, n_threads);
		dense_matrix_dense_vector_product(l_vec_hat, Xy, theta_hat, n_threads);
	}

	// Get a pointer to the columns of cr_end_pt_info
	const double *t_lb = &cr_end_pt_info[0 + start_idx];
	const double *t_ub = &cr_end_pt_info[JJ + start_idx];
	const double *f_con_der_lb = &cr_end_pt_info[4 * JJ + start_idx];
	const double *f_con_der_ub = &cr_end_pt_info[5 * JJ + start_idx];
	const double *f_con_2der_lb = &cr_end_pt_info[6 * JJ + start_idx];
	const double *f_con_2der_ub = &cr_end_pt_info[7 * JJ + start_idx];

	// Get a pointer to the correct offset in d_vec
	const double *d_vec_offset = &d_vec[start_idx];
	
	// Compute data-fit part	
	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int i = 0; i < I; i++)
		{
			double l_grp, rho = 0.0;

			const double l_old = l_vec_old[i];
			const double inv_rho = inv_rho_loc[i];

			if (inv_rho > 0)
			{
				l_grp = (l_vec[i] - l_vec_hat[i]) * inv_rho + l_old;
				rho = 1.0 / inv_rho;
			}
			else
				l_grp = l_old;

			if (rho > 0)
			{
				// Prevent exp(-x) from blowing up in -exp(-x)/(1 + exp(-x))
				if (l_grp < -700)
				{
					grad_fun_part[i] = -1;
				}
				else
				{
					double exp_neg_l, one_plus_exp;
					exp_neg_l = exp(-l_grp);
					one_plus_exp = 1 + exp_neg_l;
					grad_fun_part[i] = -(exp_neg_l / one_plus_exp);
				}
			}
		}
	}
	else
	{

		for (int i = 0; i < I; i++)
		{
			double l_grp, rho = 0.0;

			const double l_old = l_vec_old[i];
			const double inv_rho = inv_rho_loc[i];

			if (inv_rho > 0)
			{
				l_grp = (l_vec[i] - l_vec_hat[i]) * inv_rho + l_old;
				rho = 1.0 / inv_rho;
			}
			else
				l_grp = l_old;

			if (rho > 0)
			{
				// Prevent exp(-x) from blowing up in -exp(-x)/(1 + exp(-x))
				if (l_grp < -700)
				{
					grad_fun_part[i] = -1;
				}
				else
				{
					double exp_neg_l, one_plus_exp;
					exp_neg_l = exp(-l_grp);
					one_plus_exp = 1 + exp_neg_l;
					grad_fun_part[i] = -(exp_neg_l / one_plus_exp);
				}
			}
		}
	}
	
	// Dense vector matrix product
	if (mxIsSparse(Xy))
		dense_vector_sparse_matrix_product(grad, Xy, grad_fun_part, n_threads);
	else
		dense_vector_dense_matrix_product(grad, Xy, grad_fun_part, n_threads);
			
	// Include the regularization term
	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int j = 0; j < J; j++)
		{
			const double d_j = d_vec_offset[j];
			const double t_lb_j = t_lb[j];
			const double t_ub_j = t_ub[j];
			const double f_con_der_lb_j = f_con_der_lb[j];
			const double f_con_der_ub_j = f_con_der_ub[j];
			const double f_con_2der_lb_j = f_con_2der_lb[j];
			const double f_con_2der_ub_j = f_con_2der_ub[j];

			double grad_fun_reg, t_diff;

			double theta_j = theta[j];

			if (bias_term != j)
			{
				if (theta_j <= t_lb_j)
				{
					// Quadratic extension
					t_diff = theta_j - t_lb_j;

					grad_fun_reg = f_con_der_lb_j + f_con_2der_lb_j * t_diff;
				}
				else if (theta_j >= t_ub_j)
				{
					// Quadratic extension
					t_diff = theta_j - t_ub_j;

					grad_fun_reg = f_con_der_ub_j + f_con_2der_ub_j * t_diff;
				}
				else
				{
					// Convexified 1D objective
					double theta_hat_j = theta_hat[j];
					grad_fun_reg = func(theta_j, theta_hat_j, d_j);
				}

				grad[j] += grad_fun_reg;
			}
		}
	}
	else
	{
		for (int j = 0; j < J; j++)
		{
			const double d_j = d_vec_offset[j];
			const double t_lb_j = t_lb[j];
			const double t_ub_j = t_ub[j];
			const double f_con_der_lb_j = f_con_der_lb[j];
			const double f_con_der_ub_j = f_con_der_ub[j];
			const double f_con_2der_lb_j = f_con_2der_lb[j];
			const double f_con_2der_ub_j = f_con_2der_ub[j];

			double grad_fun_reg, t_diff;

			double theta_j = theta[j];

			if (bias_term != j)
			{
				if (theta_j <= t_lb_j)
				{
					// Quadratic extension
					t_diff = theta_j - t_lb_j;

					grad_fun_reg = f_con_der_lb_j + f_con_2der_lb_j * t_diff;
				}
				else if (theta_j >= t_ub_j)
				{
					// Quadratic extension
					t_diff = theta_j - t_ub_j;

					grad_fun_reg = f_con_der_ub_j + f_con_2der_ub_j * t_diff;
				}
				else
				{
					// Convexified 1D objective
					double theta_hat_j = theta_hat[j];
					grad_fun_reg = func(theta_j, theta_hat_j, d_j);
				}

				grad[j] += grad_fun_reg;
			}
		}
	}
}


/** A function for computing the product of the Hessian and a vector based on the quadratic extension
	\arg l_vec is an un-initialized vector that stores 1D projections of theta. It needs to be initialized to zeros in the code
	\arg l_vec_hat is an un-initialized vector that stores 1D projections of theta. It needs to be initialized to zeros in the code
	\arg hess_fun_part is an un-initialized vector that stores 1D second derivatives. It needs to be initialized to zeros in the code
	\xy_v is an un-initialized vector that stores the product of the feature matrix and a vector. It needs to be initialized to zeros in the code
	\arg theta is the current parameter values for the current group
	\arg v is the vector to be multiplied by the Hessian matrix implicitly (without computing the Hessian matrix)
	\arg Xy	is the feature matrix for the current group 
	\arg theta_hat is the previous parameter values for the current group
	\arg eps is the shift parameter of the penalty function
	\arg d_vec is the minimum (local range) second derivative for each parameter value
	\arg lambda is the regularization coefficient
	\arg cr_end_pt_info is a JJ x 8 matrix with end-point information for each of the parameter values
	\arg bias_term specifies the location of the bias term (if any) within the current group. A negative value signals none
	\arg l_vec_old is the vector of previous 1D projections using all the parameters
	\arg inv_rho_loc is the reciprocal of the convexifying coefficient for the current group
	\arg JJ is the total number of parameters
	\arg start_idx is the starting index of the first parameter in the current group of parameters within the entire parameter vector
	\arg n_threads is the number of threads to use for parallel processing
	\returns the Hessian times vector product in hess_vec 
	This function assumes hess_vec has already been created in memory
*/
static void hessian_vector_mult_fun_param_split(double *hess_vec, double *l_vec, double *l_vec_hat, double *hess_fun_part, double *xy_v, const double *theta, const double *v, const mxArray *Xy, const double *theta_hat, const double eps,
	const double *d_vec, const double lambda, const double *cr_end_pt_info, const int bias_term, const double *l_vec_old, const double *inv_rho_loc, 
	const int JJ, const int start_idx, const int n_threads = 0)
{
	// Initialize the functor for the 1D second derivative function
	Convexified1D::Hessian1D<double> func(eps, lambda);

	const size_t I = mxGetM(Xy);
	const size_t J = mxGetN(Xy);

	// Initialize the arrays to store the 1D projections of theta and theta_hat, the vector of 1D second derivatives, and the feature matrix and vector product to zero
	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int i = 0; i < I; i++)
		{
			l_vec[i] = 0;
			l_vec_hat[i] = 0;
			hess_fun_part[i] = 0;
			xy_v[i] = 0;
		}
	}
	else
	{
		for (int i = 0; i < I; i++)
		{
			l_vec[i] = 0;
			l_vec_hat[i] = 0;
			hess_fun_part[i] = 0;
			xy_v[i] = 0;
		}
	}

	// Compute matrix vector products
	if (mxIsSparse(Xy))
	{
		sparse_matrix_dense_vector_product(l_vec, Xy, theta, n_threads);
		sparse_matrix_dense_vector_product(l_vec_hat, Xy, theta_hat, n_threads);
		sparse_matrix_dense_vector_product(xy_v, Xy, v, n_threads);
	}
	else
	{
		dense_matrix_dense_vector_product(l_vec, Xy, theta, n_threads);
		dense_matrix_dense_vector_product(l_vec_hat, Xy, theta_hat, n_threads);
		dense_matrix_dense_vector_product(xy_v, Xy, v, n_threads);
	}

	// Get a pointer to the columns of cr_end_pt_info
	const double *t_lb = &cr_end_pt_info[0 + start_idx];
	const double *t_ub = &cr_end_pt_info[JJ + start_idx];
	const double *f_con_2der_lb = &cr_end_pt_info[6 * JJ + start_idx];
	const double *f_con_2der_ub = &cr_end_pt_info[7 * JJ + start_idx];

	// Get a pointer to the correct offset in d_vec
	const double *d_vec_offset = &d_vec[start_idx];

	// Compute data-fit part	
	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int i = 0; i < I; i++)
		{
			double l_grp, rho = 0.0;

			const double l_old = l_vec_old[i];
			const double inv_rho = inv_rho_loc[i];

			if (inv_rho > 0)
			{
				l_grp = (l_vec[i] - l_vec_hat[i]) * inv_rho + l_old;
				rho = 1.0 / inv_rho;
			}
			else
				l_grp = l_old;

			if (rho > 0)
			{
				// Prevent exp(-x) from blowing up in exp(-x)/(1 + exp(-x))^2
				if (l_grp < -700)
				{
					// Incorporate the convexity factor
					hess_fun_part[i] = 0;
				}
				else
				{
					double exp_neg_l, one_plus_exp;
					exp_neg_l = exp(-l_grp);
					one_plus_exp = 1 + exp_neg_l;

					// Incorporate the convexity factor
					hess_fun_part[i] = inv_rho * (exp_neg_l / (one_plus_exp * one_plus_exp));
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < I; i++)
		{
			double l_grp, rho = 0.0;

			const double l_old = l_vec_old[i];
			const double inv_rho = inv_rho_loc[i];

			if (inv_rho > 0)
			{
				l_grp = (l_vec[i] - l_vec_hat[i]) * inv_rho + l_old;
				rho = 1.0 / inv_rho;
			}
			else
				l_grp = l_old;

			if (rho > 0)
			{
				// Prevent exp(-x) from blowing up in exp(-x)/(1 + exp(-x))^2
				if (l_grp < -700)
				{
					// Incorporate the convexity factor
					hess_fun_part[i] = 0;
				}
				else
				{
					double exp_neg_l, one_plus_exp;
					exp_neg_l = exp(-l_grp);
					one_plus_exp = 1 + exp_neg_l;

					// Incorporate the convexity factor
					hess_fun_part[i] = inv_rho * (exp_neg_l / (one_plus_exp * one_plus_exp));
				}
			}
		}
	}

	// Multiply xy_v and hess_fun_part elementwise and save it in xy_v
	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int i = 0; i < I; i++)
			xy_v[i] = xy_v[i] * hess_fun_part[i];
	}
	else
	{
		for (int i = 0; i < I; i++)
			xy_v[i] = xy_v[i] * hess_fun_part[i];
	}

	// Dense vector matrix product
	if (mxIsSparse(Xy))
		dense_vector_sparse_matrix_product(hess_vec, Xy, xy_v, n_threads);
	else
		dense_vector_dense_matrix_product(hess_vec, Xy, xy_v, n_threads);
		
	// Include the regularization term
	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int j = 0; j < J; j++)
		{
			const double d_j = d_vec_offset[j];
			const double t_lb_j = t_lb[j];
			const double t_ub_j = t_ub[j];
			const double f_con_2der_lb_j = f_con_2der_lb[j];
			const double f_con_2der_ub_j = f_con_2der_ub[j];

			double hess_fun_reg;

			double theta_j = theta[j];

			if (bias_term != j)
			{
				if (theta_j <= t_lb_j)
				{
					// Quadratic extension
					hess_fun_reg = f_con_2der_lb_j;
				}
				else if (theta_j >= t_ub_j)
				{
					// Quadratic extension
					hess_fun_reg = f_con_2der_ub_j;
				}
				else
				{
					// Convexified 1D objective
					hess_fun_reg = func(theta_j, d_j);
				}

				hess_vec[j] += (hess_fun_reg * v[j]);
			}
		}
	}
	else
	{
		for (int j = 0; j < J; j++)
		{
			const double d_j = d_vec_offset[j];
			const double t_lb_j = t_lb[j];
			const double t_ub_j = t_ub[j];
			const double f_con_2der_lb_j = f_con_2der_lb[j];
			const double f_con_2der_ub_j = f_con_2der_ub[j];

			double hess_fun_reg;

			double theta_j = theta[j];

			if (bias_term != j)
			{
				if (theta_j <= t_lb_j)
				{
					// Quadratic extension
					hess_fun_reg = f_con_2der_lb_j;
				}
				else if (theta_j >= t_ub_j)
				{
					// Quadratic extension
					hess_fun_reg = f_con_2der_ub_j;
				}
				else
				{
					// Convexified 1D objective
					hess_fun_reg = func(theta_j, d_j);
				}

				hess_vec[j] += (hess_fun_reg * v[j]);
			}
		}
	}
}


/** A function for implementing the conjugate gradient linear system solving algorithm, where the matrix-vector product is computed without computing the entire matrix
	\arg s_opt is the solution of the trust-region linear system. The step direction
	\arg l_v is an un-initialized vector that stores 1D projections of theta.
	\arg l_v_hat is an un-initialized vector that stores 1D projections of theta.
	\arg hess_fun_part is an un-initialized vector that stores 1D second derivatives.
	\xy_v is an un-initialized vector that stores the product of the feature matrix and a vector.
	\arg r_i is an un-initialized vector of size J.
	\arg s_i is an un-initialized vector of size J. It needs to be initialized to zeros in the code
	\arg d_i is an un-initialized vector of size J.
	\arg hess_times_d_i is an un-initialized vector of size J.
	\arg s_i_plus_1 is an un-initialized vector of size J.
	\arg x_opt is the current parameter estimate at which the Hessian-vector product is to be computed
	\arg g_vec is the gradient of the objective at the current parameter estimate
	\arg A is the feature matrix
	\arg x_prev is the previous parameter estimate	
	\arg l_vec is the vector of previous 1D projections using all the parameters
	\arg d_vec is the minimum (local range) second derivative for each parameter value
	\arg cr_end_pt_info is a JJ x 8 matrix with end-point information for each of the parameter values
	\arg inv_rho_loc is the reciprocal of the convexifying coefficient for the current group
	\arg lambda is the regularization coefficient
	\arg bias_term specifies the location of the bias term (if any) within the current group. A negative value signals none
	\arg eps is the shift parameter of the penalty function
	\arg opts is struct that has fields for the conjugate gradient parameters
	\arg rad is the radius of the trust region
	\arg J is the number of parameters in s_opt, x_opt, grad, and x_prev
	\arg JJ is the total number of parameters
	\arg start_idx is the starting index of the first parameter in the current group of parameters within the entire parameter vector
	\arg n_threads is the number of threads to use for parallel processing
	\returns s_opt, the step direction
*/
static void conjugate_gradient(double *s_opt, double *l_v, double *l_v_hat, double *hess_fun_part, double *xy_v, double *r_i, double *s_i, double *d_i, double *hess_times_d_i, double *s_i_plus_1, 
	const double *x_opt, const double *g_vec, const mxArray *A, const double *x_prev, const double *l_vec,
	const double *d_vec, const double *cr_end_pt_info, const double *inv_rho_loc, const double lambda, const int bias_term, const double eps, 
	const Options opts, const double rad, const int J, const int JJ, const int start_idx, const int n_threads = 0)
{
	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int j = 0; j < J; j++)
		{
			// Initialize s_i to zero
			s_i[j] = 0; 

			r_i[j] = -g_vec[j];
			d_i[j] = r_i[j];
		}
	}
	else
	{
		for (int j = 0; j < J; j++)
		{
			// Initialize s_i to zero
			s_i[j] = 0;

			r_i[j] = -g_vec[j];
			d_i[j] = r_i[j];
		}
	}

	double grad_tol = opts.cg_xi * l2_norm(g_vec, J);

	for (int i = 0; i < opts.cg_max_iter; i++)
	{
		double norm_r_i = l2_norm(r_i, J);
		
		if (norm_r_i <= grad_tol)
		{
			if (n_threads > 0)
			{
				#pragma omp parallel for default(shared) num_threads(n_threads)
				for (int j = 0; j < J; j++)
					s_opt[j] = s_i[j];				
			}
			else
			{
				for (int j = 0; j < J; j++)
					s_opt[j] = s_i[j];				
			}

			if (opts.display)
				mexPrintf("CG: Optimality condition for gradient value, grad_tol, reached\n");

			return;
		}

		hessian_vector_mult_fun_param_split(hess_times_d_i, l_v, l_v_hat, hess_fun_part, xy_v, x_opt, d_i, A, x_prev, eps, d_vec, lambda, cr_end_pt_info, bias_term, l_vec, inv_rho_loc, JJ, start_idx, n_threads);

		double norm_r_i_sq = norm_r_i * norm_r_i;
		
		double alpha_i = norm_r_i_sq / inner_product(d_i, hess_times_d_i, J);

		if (n_threads > 0)
		{
			#pragma omp parallel for default(shared) num_threads(n_threads)
			for (int j = 0; j < J; j++)
				s_i_plus_1[j] = s_i[j] + alpha_i * d_i[j]; 
		}
		else
		{
			for (int j = 0; j < J; j++)
				s_i_plus_1[j] = s_i[j] + alpha_i * d_i[j];
		}

		if (l2_norm(s_i_plus_1, J) >= rad)
		{
			// Find tau such that || s_i + tau * d_i || = rad
			double a = inner_product(d_i, d_i, J);
			double b = 2 * inner_product(s_i, d_i, J);
			double c = inner_product(s_i, s_i, J) - (rad * rad);

			// Find the biggest root of the quadratic
			double rt_1, rt_2;
			double D = sqrt(b * b - 4 * a * c);
			double two_a = 2 * a;

			if (a < 0 || a > 0)
			{
				rt_1 = (-b + D) / two_a;
				rt_2 = (-b - D) / two_a;
			}
			else
			{
				rt_1 = -c / b;
				rt_2 = rt_1;
			}
			double tau = std::max(rt_1, rt_2);

			// Out of curiosity, display error whenever tau >= alpha_i or tau is not positive
			if (tau >= alpha_i || tau <= 0)
			{
				mexPrintf("i = %d, norm_s_i_1 = %f, norm_s_i = %f, norm_d_i = %f, alpha_i = %f, rad = %f, a = %f, b = %f, c = %f, rt_1 = %f, rt_2 = %f, tau = %f\n", 
					i, l2_norm(s_i_plus_1, J), l2_norm(s_i, J), l2_norm(d_i, J), alpha_i, rad, a, b, c, rt_1, rt_2, tau);
				mexPrintf("The stepsize in the conjugate gradient inner loop is meaningless because the Hessian is not PD\n");
				mexPrintf("Setting stepsize to 0\n");
				tau = 0;
			}

			if (n_threads > 0)
			{
				#pragma omp parallel for default(shared) num_threads(n_threads)
				for (int j = 0; j < J; j++)
					s_opt[j] = s_i[j] + tau * d_i[j];
			}
			else
			{
				for (int j = 0; j < J; j++)
					s_opt[j] = s_i[j] + tau * d_i[j];
			}
		
			if (opts.display)
				mexPrintf("CG: Optimality condition for trust region radius\n");
			
			return;			
		}

		if (n_threads > 0)
		{
			#pragma omp parallel for default(shared) num_threads(n_threads)
			for (int j = 0; j < J; j++)
			{
				s_i[j] = s_i_plus_1[j];
				r_i[j] = r_i[j] - alpha_i * hess_times_d_i[j];
			}			
		}
		else
		{
			for (int j = 0; j < J; j++)
			{
				s_i[j] = s_i_plus_1[j];
				r_i[j] = r_i[j] - alpha_i * hess_times_d_i[j];
			}
		}		

		double norm_r_i_new = l2_norm(r_i, J);

		double beta_i = (norm_r_i_new * norm_r_i_new) / norm_r_i_sq;

		if (n_threads > 0)
		{
			#pragma omp parallel for default(shared) num_threads(n_threads)
			for (int j = 0; j < J; j++)
			{
				d_i[j] = r_i[j] + beta_i * d_i[j];
			}
		}
		else
		{
			for (int j = 0; j < J; j++)
			{
				d_i[j] = r_i[j] + beta_i * d_i[j];
			}
		}		
	}

	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int j = 0; j < J; j++)
		{
			s_opt[j] = s_i[j];
		}
	}
	else
	{
		for (int j = 0; j < J; j++)
		{
			s_opt[j] = s_i[j];
		}
	}
}


/** A function for implementing the trust-region method based on Conjugate gradient 
	\arg l_v is an un-initialized vector that stores 1D projections of theta.
	\arg l_v_hat is an un-initialized vector that stores 1D projections of theta. 
	\arg grad_fun_part is an un-initialized vector that stores 1D gradients.
	\arg hess_fun_part is an un-initialized vector that stores 1D second derivatives. 
	\xy_v is an un-initialized vector that stores the product of the feature matrix and a vector.
	\arg r_i is an un-initialized vector of size J.
	\arg s_i is an un-initialized vector of size J.
	\arg d_i is an un-initialized vector of size J.
	\arg hess_times_d_i is an un-initialized vector of size J.
	\arg s_i_plus_1 is an un-initialized vector of size J.
	\arg grad is an un-initialized vector of size J.
	\arg s_opt is an un-initialized vector of size J.
	\arg x_new is an un-initialized vector of size J.
	\arg hv is an un-initialized vector of size J.
	\arg x_prev is the previous estimate and initial starting point. A J-vector
	\arg A is the feature matrix 
	\arg l_vec is the vector of previous 1D projections using all the parameters
	\arg d_vec is the minimum (local range) second derivative for each parameter value
	\arg cr_end_pt_info is a JJ x 8 matrix with end-point information for each of the parameter values
	\arg inv_rho_loc is the reciprocal of the convexifying coefficient for the current group
	\arg lambda is the regularization coefficient
	\arg bias_term specifies the location of the bias term (if any) within the current group. A negative value signals none
	\arg eps is the shift parameter of the penalty function
	\arg opt is struct that has fields for the trust region parameters
	\arg J is the number of parameters in x_opt and x_prev
	\arg JJ is the total number of parameters
	\arg start_idx is the starting index of the first parameter in the current group of parameters within the entire parameter vector
	\arg n_threads is the number of threads to use for parallel processing
	\returns x_opt, the optimized point 
*/
static void trust_region_ND_CG_unbounded(double *x_opt, double *l_v, double *l_v_hat, double *grad_fun_part, double *hess_fun_part, double *xy_v,
	double *r_i, double *s_i, double *d_i, double *hess_times_d_i, double *s_i_plus_1, double *grad, double *s_opt, double *x_new, double *hv, 
	const double *x_prev, const mxArray *A, const double *l_vec, const double *d_vec, const double *cr_end_pt_info, const double *inv_rho_loc, 
	const double lambda, const int bias_term, const double eps, const Options opts, const int J, const int JJ, const int start_idx, const int n_threads = 0)
{
	// Initialize the output
	if (n_threads > 0)
	{
		#pragma omp parallel for default(shared) num_threads(n_threads)
		for (int j = 0; j < J; j++)
			x_opt[j] = x_prev[j];
	}
	else
	{
		for (int j = 0; j < J; j++)
			x_opt[j] = x_prev[j];
	}
	
	grad_fun_param_split(grad, l_v, l_v_hat, grad_fun_part, x_opt, A, x_prev, eps, d_vec, lambda, cr_end_pt_info, bias_term, l_vec, inv_rho_loc, JJ, start_idx, n_threads);
	
	// Initialize the trust region radius
	double rad = l2_norm(grad, J);

	// Initialize objective function value
	double fun_val_k; 
	obj_fun_param_split(fun_val_k, l_v, l_v_hat, x_opt, A, x_prev, eps, d_vec, lambda, cr_end_pt_info, bias_term, l_vec, inv_rho_loc, JJ, start_idx, n_threads);
		
	for (int k = 0; k < opts.max_iter; k++)
	{
		grad_fun_param_split(grad, l_v, l_v_hat, grad_fun_part, x_opt, A, x_prev, eps, d_vec, lambda, cr_end_pt_info, bias_term, l_vec, inv_rho_loc, JJ, start_idx, n_threads);

		double norm_g_vec = l2_norm(grad, J);

		if (norm_g_vec < opts.grad_tol)
		{
			if (opts.display)
				mexPrintf("TR: Optimality condition for gradient norm, grad_tol, reached\n");

			break;
		}

		conjugate_gradient(s_opt, l_v, l_v_hat, hess_fun_part, xy_v, r_i, s_i, d_i, hess_times_d_i, s_i_plus_1, x_opt, grad, A, x_prev, l_vec, d_vec, cr_end_pt_info,
			inv_rho_loc, lambda, bias_term, eps, opts, rad, J, JJ, start_idx, n_threads);

		if (opts.display)
			mexPrintf("TR: Delta_k, norm(s_opt) %f, %f\n", rad, l2_norm(s_opt, J));

		if (n_threads > 0)
		{
			#pragma omp parallel for default(shared) num_threads(n_threads)
			for (int j = 0; j < J; j++)
				x_new[j] = x_opt[j] + s_opt[j];
		}
		else
		{
			for (int j = 0; j < J; j++)
				x_new[j] = x_opt[j] + s_opt[j];
		}

		// Compute the actual and predicted function values
		double old_fun_val_k, new_fun_val_k, act_red_k, pred_red_k, ratio_k;
		obj_fun_param_split(new_fun_val_k, l_v, l_v_hat, x_new, A, x_prev, eps, d_vec, lambda, cr_end_pt_info, bias_term, l_vec, inv_rho_loc, JJ, start_idx, n_threads);
		act_red_k = new_fun_val_k - fun_val_k;
		hessian_vector_mult_fun_param_split(hv, l_v, l_v_hat, hess_fun_part, xy_v, x_opt, s_opt, A, x_prev, eps, d_vec, lambda, cr_end_pt_info, bias_term, l_vec, inv_rho_loc, JJ, start_idx, n_threads);
		pred_red_k = inner_product(grad, s_opt, J) + 0.5 * inner_product(s_opt, hv, J);  // should be negative!

		ratio_k = act_red_k / pred_red_k;

		// Update x_opt
		old_fun_val_k = fun_val_k;
		if (ratio_k > opts.eta_0)
		{
			if (n_threads > 0)
			{
				#pragma omp parallel for default(shared) num_threads(n_threads)
				for (int j = 0; j < J; j++)
					x_opt[j] = x_new[j];
			}
			else
			{
				for (int j = 0; j < J; j++)
					x_opt[j] = x_new[j];
			}

			fun_val_k = new_fun_val_k;
		}

		// Update rad
		// Choose rad as gamma_k * || s_k || , where gamma_k is the minimum of
		// a quadratic that interpolates the function phi(gamma) = f_fun(x_k + gamma * s_k).
		// This quadratic satisfies phi(0) = f_fun(x_k), phi'(0) = g_fun(x_k)' * s_k, and phi(1) = f_fun(x_k + s_k).
		// If phi(gamma) does not have a minimum, set gamma_k = +Inf.
		// Choose rad as gamma_k * || s_k || , if it falls in the desired interval[rad_lb, rad_ub].
		// Otherwise, set rad to the closest endpoint to gamma_k.
		double norm_s_k, rad_lb, rad_ub, q_a, q_b, q_c, gamma_k, inf;
		inf = std::numeric_limits<double>::infinity();

		norm_s_k = l2_norm(s_opt, J);
		if (ratio_k <= opts.eta_1)
		{
			rad_lb = opts.sigma_1 * std::min(norm_s_k, rad);
			rad_ub = opts.sigma_2 * rad;
		}
		else if (ratio_k < opts.eta_2)
		{
			rad_lb = opts.sigma_1 * rad;
			rad_ub = opts.sigma_3 * rad;
		}
		else
		{
			rad_lb = rad;
			rad_ub = opts.sigma_3 * rad;
		}

		// Find the minimum point of phi(gamma) = f_fun(x_k + gamma * s_k)
		q_b = inner_product(grad, s_opt, J);
		q_c = old_fun_val_k;
		q_a = new_fun_val_k - q_b - q_c;

		if (q_a > 0)
			gamma_k = -q_b / (2 * q_a);
		else
			gamma_k = inf;
		
		rad = std::min(std::max(gamma_k * norm_s_k, rad_lb), rad_ub);

		// If the actual function reduction is small, exit
		if (fabs(act_red_k) < opts.fun_tol)
		{
			if (opts.display)
				mexPrintf("TR: Optimality condition for function value, fun_tol, reached\n");

			break;
		}

		// If the trust region size is too small, exit
		if (fabs(rad) < opts.rad_tol)
		{
			if (opts.display)
				mexPrintf("TR: Optimality condition for trust region radius, rad_tol, reached\n");

			break;
		}

	}
}


// The main computational engine
// 1 output, 18 inputs. Wow!
static void PMM_CG_nonconvex_logistic_regression(double *theta_out, const mxArray *Xy_grp, const double *theta, const double *grp_splits, const double *cr_end_pt_info, const double *l_vec,
						const double *d_vec, const double lambda, const double *inv_rho_grp, const int bias_grp, const int bias_idx, const double eps, 
						const mxArray *options, const int K, const int I, const int JJ, const int num_outer_threads, const int num_inner_threads)
{
	// I chose to make grp_splits doubles instead of int32s, because MATLAB uses doubles by default.
	// Its values will be type cast into ints before being used
	
	// The matrices cr_end_pt_info and inv_rho_grp are stored by MATLAB in column-major order. That is, the entries of each column are contiguous. 
	// Accessing contiguous memory is more efficient than not doing so

	// Get the algorithm options from the MATLAB struct and create C/C++ struct
	// The fields are: eta_0, eta_1, eta_2, sigma_1, sigma_2, sigma_3, max_iter, fun_tol, grad_tol, rad_tol, cg_xi_k, cg_max_iter, display
	Options opts;

	int field_num;

	field_num = mxGetFieldNumber(options, "eta_0");
	if (field_num >= 0)
		opts.eta_0 = mxGetScalar(mxGetFieldByNumber(options, 0, field_num));

	field_num = mxGetFieldNumber(options, "eta_1");
	if (field_num >= 0)
		opts.eta_1 = mxGetScalar(mxGetFieldByNumber(options, 0, field_num));

	field_num = mxGetFieldNumber(options, "eta_2");
	if (field_num >= 0)
		opts.eta_2 = mxGetScalar(mxGetFieldByNumber(options, 0, field_num));

	field_num = mxGetFieldNumber(options, "sigma_1");
	if (field_num >= 0)
		opts.sigma_1 = mxGetScalar(mxGetFieldByNumber(options, 0, field_num));

	field_num = mxGetFieldNumber(options, "sigma_2");
	if (field_num >= 0)
		opts.sigma_2 = mxGetScalar(mxGetFieldByNumber(options, 0, field_num));

	field_num = mxGetFieldNumber(options, "sigma_3");
	if (field_num >= 0)
		opts.sigma_3 = mxGetScalar(mxGetFieldByNumber(options, 0, field_num));
	
	field_num = mxGetFieldNumber(options, "max_iter");
	if (field_num >= 0)
		opts.max_iter = (int) mxGetScalar(mxGetFieldByNumber(options, 0, field_num));

	field_num = mxGetFieldNumber(options, "fun_tol");
	if (field_num >= 0)
		opts.fun_tol = mxGetScalar(mxGetFieldByNumber(options, 0, field_num));

	field_num = mxGetFieldNumber(options, "grad_tol");
	if (field_num >= 0)
		opts.grad_tol = mxGetScalar(mxGetFieldByNumber(options, 0, field_num));
	
	field_num = mxGetFieldNumber(options, "rad_tol");
	if (field_num >= 0)
		opts.rad_tol = mxGetScalar(mxGetFieldByNumber(options, 0, field_num));

	field_num = mxGetFieldNumber(options, "cg_xi_k");
	if (field_num >= 0)
		opts.cg_xi = mxGetScalar(mxGetFieldByNumber(options, 0, field_num));

	field_num = mxGetFieldNumber(options, "cg_max_iter");
	if (field_num >= 0)
		opts.cg_max_iter = (int) mxGetScalar(mxGetFieldByNumber(options, 0, field_num));

	field_num = mxGetFieldNumber(options, "display");
	if (field_num >= 0)
		opts.display = (int) mxGetScalar(mxGetFieldByNumber(options, 0, field_num));
	
	/*******************************************************
	Memory allocation across parameter groups
	********************************************************/
	// Get the maximum number of parameters across all groups
	size_t J_max = 0;
	for (int k = 0; k < K; ++k)
	{
		// Get the matrix in the k-th cell of Xy_grp
		const mxArray *Xy_loc = mxGetCell(Xy_grp, k);

		// Number of parameters in current group
		const size_t J = mxGetN(Xy_loc);

		J_max = std::max(J_max, J);
	}

	// Allocate all the arrays needed for each helper function call within each parameter group (possibly threads)
	double *l_v, *l_v_hat, *grad_fun_part, *hess_fun_part, *xy_v, *r_i, *s_i, *d_i, *hess_times_d_i, *s_i_plus_1, *grad, *s_opt, *x_new, *hv;
		
	l_v = (double *) malloc(K * I * sizeof(double));
	l_v_hat = (double *) malloc(K * I * sizeof(double));
	grad_fun_part = (double *) malloc(K * I * sizeof(double));
	hess_fun_part = (double *) malloc(K * I * sizeof(double));
	xy_v = (double *) malloc(K * I * sizeof(double));
	r_i = (double *) malloc(K * J_max * sizeof(double));
	s_i = (double *) malloc(K * J_max * sizeof(double));
	d_i = (double *) malloc(K * J_max * sizeof(double));
	hess_times_d_i = (double *) malloc(K * J_max * sizeof(double));
	s_i_plus_1 = (double *) malloc(K * J_max * sizeof(double));
	grad = (double *) malloc(K * J_max * sizeof(double));
	s_opt = (double *) malloc(K * J_max * sizeof(double));
	x_new = (double *) malloc(K * J_max * sizeof(double));
	hv = (double *) malloc(K * J_max * sizeof(double));	
				
	if (num_outer_threads > 0)
	{
		// Loop over the parameter groups		
		#pragma omp parallel for default(shared) num_threads(num_outer_threads)
		for (int k = 0; k < K; ++k)
		{	
			// Index of the bias term in the group of parameters
			int bias_term = -1;

			if (bias_grp == k)
				bias_term = bias_idx;

			// Get the matrix in the k-th cell of Xy_grp
			const mxArray *Xy_loc = mxGetCell(Xy_grp, k);

			// Get a pointer to the k-th col of inv_rho_grp (column-major indexing)
			const double *inv_rho_loc = &inv_rho_grp[I * k];

			// Get the index to the starting location of the parameter set
			int param_start_idx = (int) grp_splits[k];

			const double *theta_loc = &theta[param_start_idx];

			double *theta_out_loc = &theta_out[param_start_idx];

			// Number of parameters in current group
			const size_t J = mxGetN(Xy_loc);
			
			// Get the k-th row of each temporary array
			double *l_v_loc = &l_v[I * k];
			double *l_v_hat_loc = &l_v_hat[I * k];
			double *grad_fun_part_loc = &grad_fun_part[I * k];
			double *hess_fun_part_loc = &hess_fun_part[I * k];
			double *xy_v_loc = &xy_v[I * k];
			double *r_i_loc = &r_i[J_max * k];
			double *s_i_loc = &s_i[J_max * k];
			double *d_i_loc = &d_i[J_max * k];
			double *hess_times_d_i_loc = &hess_times_d_i[J_max * k];
			double *s_i_plus_1_loc = &s_i_plus_1[J_max * k];
			double *grad_loc = &grad[J_max * k];
			double *s_opt_loc = &s_opt[J_max * k];
			double *x_new_loc = &x_new[J_max * k];
			double *hv_loc = &hv[J_max * k];
						
			// Call the trust-region method based on conjugate gradient
			trust_region_ND_CG_unbounded(theta_out_loc, l_v_loc, l_v_hat_loc, grad_fun_part_loc, hess_fun_part_loc, xy_v_loc, r_i_loc, 
				s_i_loc, d_i_loc, hess_times_d_i_loc, s_i_plus_1_loc, grad_loc, s_opt_loc, x_new_loc, hv_loc, 
				theta_loc, Xy_loc, l_vec, d_vec, cr_end_pt_info, inv_rho_loc, lambda, bias_term, eps, opts, J, JJ, param_start_idx, num_inner_threads);			
		}
	}
	else
	{
		double *l_v_loc, *l_v_hat_loc, *grad_fun_part_loc, *hess_fun_part_loc, *xy_v_loc, *r_i_loc, *s_i_loc,
			*d_i_loc, *hess_times_d_i_loc, *s_i_plus_1_loc, *grad_loc, *s_opt_loc, *x_new_loc, *hv_loc;

		// Loop over the parameter groups
		for (int k = 0; k < K; ++k)
		{
			// Index of the bias term in the group of parameters
			int bias_term = -1;

			if (bias_grp == k)
				bias_term = bias_idx;

			// Get the matrix in the k-th cell of Xy_grp
			const mxArray *Xy_loc = mxGetCell(Xy_grp, k);

			// Get a pointer to the k-th col of inv_rho_grp (column-major indexing)
			const double *inv_rho_loc = &inv_rho_grp[I * k];

			// Get the index to the starting location of the parameter set
			int param_start_idx = (int)grp_splits[k];

			const double *theta_loc = &theta[param_start_idx];

			double *theta_out_loc = &theta_out[param_start_idx];

			// Number of parameters in current group
			const size_t J = mxGetN(Xy_loc);
			
			// Get the k-th row of each temporary array
			l_v_loc = &l_v[I * k];
			l_v_hat_loc = &l_v_hat[I * k];
			grad_fun_part_loc = &grad_fun_part[I * k];
			hess_fun_part_loc = &hess_fun_part[I * k];
			xy_v_loc = &xy_v[I * k];
			r_i_loc = &r_i[J_max * k];
			s_i_loc = &s_i[J_max * k];
			d_i_loc = &d_i[J_max * k];
			hess_times_d_i_loc = &hess_times_d_i[J_max * k];
			s_i_plus_1_loc = &s_i_plus_1[J_max * k];
			grad_loc = &grad[J_max * k];
			s_opt_loc = &s_opt[J_max * k];
			x_new_loc = &x_new[J_max * k];
			hv_loc = &hv[J_max * k];
			
			// Call the trust-region method based on conjugate gradient
			trust_region_ND_CG_unbounded(theta_out_loc, l_v_loc, l_v_hat_loc, grad_fun_part_loc, hess_fun_part_loc, xy_v_loc, r_i_loc,
				s_i_loc, d_i_loc, hess_times_d_i_loc, s_i_plus_1_loc, grad_loc, s_opt_loc, x_new_loc, hv_loc, 
				theta_loc, Xy_loc, l_vec, d_vec, cr_end_pt_info, inv_rho_loc, lambda, bias_term, eps, opts, J, JJ, param_start_idx, num_inner_threads);
		}
	}
	
	// Free temporary allocated memory
	free(l_v);
	free(l_v_hat); 
	free(grad_fun_part); 
	free(hess_fun_part); 
	free(xy_v); 
	free(r_i); 
	free(s_i); 
	free(d_i); 
	free(hess_times_d_i); 
	free(s_i_plus_1); 
	free(grad); 
	free(s_opt); 
	free(x_new); 
	free(hv);	
}


// Gateway (mex) routine
// Function signature is always the same
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	// Check for proper number of arguments
	if(nlhs > 1)
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:nargout", "PMM_CG_nonconvex_logistic_regression requires at most 1 output argument."); 

	if(nrhs < 12 || nrhs > 14)
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:nargin", "PMM_CG_nonconvex_logistic_regression requires at least 12 input arguments and at most 14.");
	
	// Check for input data type mismatch
	if (!IS_CELL_ARRAY_OF_REAL_MATRIX(prhs[0]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input Xy_grp must be a cell array of matrices.");

	if (!IS_REAL_1D_FULL_DOUBLE(prhs[1]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input theta must be a real, full, vector of doubles.");
	
	if (!IS_REAL_1D_FULL_DOUBLE(prhs[2]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input grp_splits must be a real, full, vector of doubles.");

	if (!IS_REAL_2D_FULL_DOUBLE(prhs[3]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input cr_end_pt_info must be a real, full, 2D matrix of doubles.");
	
	if (!IS_REAL_1D_FULL_DOUBLE(prhs[4]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input l_vec must be a real, full, vector of doubles.");

	if (!IS_REAL_1D_FULL_DOUBLE(prhs[5]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input d_vec must be a real, full, vector of doubles.");

	if (!IS_REAL_SCALAR_DOUBLE(prhs[6]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input lambda must be a scalar double.");

	if (!IS_REAL_2D_FULL_DOUBLE(prhs[7]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input inv_rho_grp must be a real, full, 2D matrix of doubles.");

	if (!IS_REAL_SCALAR_DOUBLE(prhs[8]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input bias_grp must be a scalar double.");
	
	if (!IS_REAL_SCALAR_DOUBLE(prhs[9]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input bias_idx must be a scalar double.");

	if (!IS_REAL_SCALAR_DOUBLE(prhs[10]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input eps must be a scalar double.");

	if (!IS_STRUCT(prhs[11]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input options must be a structure.");
	
	if(nrhs > 12)
	{
		if (!IS_REAL_SCALAR_DOUBLE(prhs[12]))
			mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input num_outer_threads must be a scalar double.");

		if (nrhs > 13)
		{
			if (!IS_REAL_SCALAR_DOUBLE(prhs[13]))
				mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:type_mismatch", "Input num_inner_threads must be a scalar double.");
		}
	}

	// Check for dimension mismatch amongst the input variables	
	int K = (int) mxGetNumberOfElements(prhs[0]); // The number of cells in the cell array of matrices. This matches the number of partition
	int I = (int) mxGetM(mxGetCell(prhs[0], 0)); // Each matrix will have I rows. Too lazy to check this for all the matrices
	int J = (int) mxGetNumberOfElements(prhs[1]); // The total number of parameters
				
	if (K + 1 != mxGetNumberOfElements(prhs[2]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:dimension_mismatch", "The number of partitions + 1 must match the number of elements in grp_splits.");

	if (J != mxGetM(prhs[3]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:dimension_mismatch", "The number of parameters must match the number of rows in cr_end_pt_info.");

	if (8 != mxGetN(prhs[3]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:dimension_mismatch", "The number of columns in cr_end_pt_info must be 8.");

	if (I != mxGetNumberOfElements(prhs[4]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:dimension_mismatch", "The number of rows in the matrix of each cell of Xy_grp must match the number of elements in l_vec.");

	if (J != mxGetNumberOfElements(prhs[5]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:dimension_mismatch", "The number of parameters must match the number of elements in d_vec.");

	if (I != mxGetM(prhs[7]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:dimension_mismatch", "The number of rows in the matrix of each cell of Xy_grp must match the number of rows in inv_rho_grp.");

	if (K != mxGetN(prhs[7]))
		mexErrMsgIdAndTxt("MATLAB:PMM_CG_nonconvex_logistic_regression:dimension_mismatch", "The number of partitions must match the number of columns in inv_rho_grp.");
	

	// Create the output vector. The output vector is already initialized to 0, on creation
	plhs[0] = mxCreateDoubleMatrix(J, 1, mxREAL);
	
	// Get the pointer to the output array
	double *theta_out = mxGetPr(plhs[0]);

	// Get the pointers to the input arrays 
	const double *theta = mxGetPr(prhs[1]);
	const double *grp_splits = mxGetPr(prhs[2]);
	const double *cr_end_pt_info = mxGetPr(prhs[3]);
	const double *l_vec = mxGetPr(prhs[4]);
	const double *d_vec = mxGetPr(prhs[5]);
	const double lambda = mxGetScalar(prhs[6]);
	const double *inv_rho_grp = mxGetPr(prhs[7]);
	const int bias_grp = (int) mxGetScalar(prhs[8]);
	const int bias_idx = (int) mxGetScalar(prhs[9]);
	const double eps = mxGetScalar(prhs[10]);
	int num_outer_threads, num_inner_threads;

	if (nrhs < 13)
		num_outer_threads = (int) std::min((double)K, 32.0); // Set the number of threads equal to the number of groups, if unspecified
	else
		num_outer_threads = (int) mxGetScalar(prhs[12]);

	if (nrhs < 14)
		num_inner_threads = 0;
	else
		num_inner_threads = (int) mxGetScalar(prhs[13]);

	// OpenMP set up
	#if ENABLE_OMP
	// omp_set_num_threads(n_threads); // The number of threads will be set by the loops as needed
	omp_set_dynamic(0); // 0 -> prevents the number of threads available in subsequent parallel region from being adjusted by the run time
	omp_set_nested(1); // 1 -> allows nested parallel regions
	#endif

	// Calculate the number of outer and inner threads (nested threads)
	// The number of outer threads should equal the number of groups (partitions) ideally, unless it exceeds 
	// the available allowed number of threads
	// The number of inner threads (per outer thread) should satisfy the inequality
	// (number of outer threads x number of inner threads) + number of outer threads <= number of threads
	// const int num_outer_threads = (int) std::min((double)K, (double)n_threads);
	// const int num_inner_threads = (n_threads - num_outer_threads) / num_outer_threads;
	
	// Call the computational routine
	PMM_CG_nonconvex_logistic_regression(theta_out, prhs[0], theta, grp_splits, cr_end_pt_info, l_vec, d_vec, lambda,  
		inv_rho_grp, bias_grp, bias_idx, eps, prhs[11], K, I, J, num_outer_threads, num_inner_threads);
	
	return;
}
