/*
 * Matrix.h
 *
 *  Created on: 18-Dec-2021
 *      Author: gnuchessmen
 */
#include <vector>

#ifndef MATRIX_H_
#define MATRIX_H_

namespace mat {

class Matrix {
protected:
	std:: vector<std:: vector<double> > matrix;

public:
	// Is the vector-in-vector a valid matrix?
	bool is_matrix(std:: vector<std:: vector<double> > matrix);

	// Matrix setter
	void mat_def(std:: vector<std:: vector<double> > A);

	// Number of rows (getter)
	int N_rows();

	// Number of columns (getter)
	int N_cols();

	// Returning the vector in vector (getter)
	std:: vector<std:: vector<double> > mat_get();

	// Printing the matrix (getter)
	void mat_print();

	// Individual element (getter)
	double element(int i, int j);

	// Individual element (setter)
	void element(int i, int j, double u);

	// Matrix multiplication
	Matrix operator* (const Matrix& other);

	// Determinant of a matrix
	double det();

	// Transpose of matrix
	Matrix t();
};


// Scalar multiplication 1
Matrix operator* (Matrix Z, const double other);

// Scalar multiplication 2
Matrix operator* (const double other, Matrix Z);

// Scalar division
Matrix operator/ (Matrix Z, const double other);

// Exponential
Matrix operator^ (Matrix Z, const int other);

// Matrix Addition
Matrix operator+ (Matrix Y, Matrix Z);

// Matrix Subtraction
Matrix operator- (Matrix Y, Matrix Z);

// Unary matrix negation
Matrix operator- (Matrix Z);

// Norm of matrix
double norm(Matrix _A);

// Identity matrix generator function
Matrix Identity(int n);

// Zeros matrix generator function
Matrix Zeros(int n, int m);

// Ones matrix generator function
Matrix Ones(int n, int m);

// Solving a system of linear equations with Naive Gaussian elimination
Matrix naiveGaussianSolver(Matrix _A, Matrix _b);

// Solving a system of linear equations with Gauss-Seidel method
Matrix GaussSeidel_SOR_Solver(Matrix _A, Matrix _b, double w, double tol);

// Finding the dominant eigenvector from Power method
Matrix PowerMethod(Matrix _A, double tol);

// Finding the other eigenvectors from Inverse Power method
Matrix InversePowerMethod(Matrix _A, double shift, double tol);

} /* namespace mat */

#endif /* MATRIX_H_ */
