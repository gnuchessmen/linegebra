/*
 * Matrix.cpp
 *
 *  Created on: 18-Dec-2021
 *      Author: gnuchessmen
 */
#include <iostream>
#include <vector>
#include <cmath>
#include "Matrix.h"

namespace mat {
// Is the vector-in-vector a valid matrix?
bool Matrix::is_matrix(std:: vector<std:: vector<double> > matrix){
	int num_rows = matrix.size();
	int num_columns = matrix[0].size();

	for(int i = 0; i < num_rows; i++){
		if(num_columns != matrix[i].size()){
			return false;
		}
		else{
			continue;
		}
	}
	return true;
}

// Matrix setter
void Matrix::mat_def(std:: vector<std:: vector<double> > A){
	if(is_matrix(A) == true){
		matrix = A;
	}
	else{
		std:: cout << "Invalid matrix!" << std:: endl;
	}
}

// Number of rows (getter)
int Matrix::N_rows(){
	return matrix.size();
}

// Number of columns (getter)
int Matrix::N_cols(){
	return matrix[0].size();
}

// Returning the vector in vector (getter)
std:: vector<std:: vector<double> > Matrix::mat_get(){
	return this->matrix;
}

// Printing the matrix (getter)
void Matrix::mat_print(){
	int num_rows = matrix.size();
	int num_columns = matrix[0].size();

	std:: cout << "(" << num_rows << " x " << num_columns << " matrix)\n" << std:: endl;

	if(num_rows == 0){
		std:: cout << "Empty matrix!" << std:: endl;
		return;
	}

	for(int i = 0; i < num_rows; i++){
		for(int j = 0; j < num_columns; j++){
			std:: cout << "\t" << matrix[i][j];
		}
		std:: cout << std:: endl;
	}

	std:: cout << std:: endl;

	return;
}

// Individual element (getter)
double Matrix::element(int i, int j){
	return matrix[i][j];
}

// Individual element (setter)
void Matrix::element(int i, int j, double u){
	matrix[i][j] = u;
	return;
}

// Matrix multiplication
Matrix Matrix::operator* (const Matrix& other){
	std:: vector<std:: vector<double> > A = this->matrix;
	std:: vector<std:: vector<double> > B = other.matrix;

	if(A[0].size() != B.size()){
		std:: cout << "Number of columns of the matrix_1 is not equal to number of rows matrix_2" << std:: endl;
		// std:: vector<std:: vector<double>> EMPTY = {};
		Matrix EMPTY;
		// EMPTY.mat_def({{}});
		return EMPTY;
	}
	else{
		std:: vector<std:: vector<double> > C;
		std:: vector<double> TEMP;

		double sum = 0;
		for(int y = 0; y < A.size(); y++){
			for(int z = 0; z < B[0].size(); z++){
				for(int i = 0; i < A[0].size(); i++){
					sum = sum + A[y][i]*B[i][z];
				}
				TEMP.push_back(sum);
				sum = 0;
			}
			C.push_back(TEMP);
			TEMP.clear();
		}
		Matrix D;
		D.mat_def(C);

		return D;
	}
}


// Determinant of a matrix
double Matrix::det(){
	if(matrix.size() == 2){
		return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0];
	}
	else{
		double sum = 0;
		// for(int i = 0; i < matrix.size(); i++){
		int i = 0;
		for(int j = 0; j < matrix.size(); j++){
			std:: vector<std:: vector<double> > new_matrix;
			std:: vector<double> TEMP;
			for(int x = 0; x < matrix.size(); x++){
				for(int y = 0; y < matrix.size(); y++){
					if((x != i) && (y != j)){
						TEMP.push_back(matrix[x][y]);
					}
				}
				if(TEMP.empty() == false){
					new_matrix.push_back(TEMP);
					TEMP.clear();
				}
			}
			Matrix new_INPUT;
			new_INPUT.mat_def(new_matrix);
			// new_INPUT.mat_print();

			if((i+j) % 2 == 0){
				sum = sum + matrix[i][j]*new_INPUT.det();
			}
			else{
				sum = sum - matrix[i][j]*new_INPUT.det();
			}

			// std:: cout << sum << std:: endl;
			// std:: cout << new_INPUT.det() << std:: endl;
			// std:: cout << matrix[i][j] << std:: endl;
			// std:: cout << matrix[i][j]*new_INPUT.det() << std:: endl;
			// std:: cout << (i+j) % 2 << std:: endl;

		}
		// }
		return sum;
	}
}

// Transpose of matrix
Matrix Matrix::t(){
	std:: vector<std:: vector<double> > A = this->matrix;
	std:: vector<std:: vector<double> > B;
	std:: vector<double> TEMP;

	for(int i = 0; i < A[0].size(); i++){
		for(int j = 0; j < A.size(); j++){
			TEMP.push_back(A[j][i]);
		}
		B.push_back(TEMP);
		TEMP.clear();
	}

	Matrix C;
	C.mat_def(B);

	return C;
}

// Scalar multiplication 1
Matrix operator* (Matrix Z, const double other){
	std:: vector<std:: vector<double> > A = Z.mat_get();
	std:: vector<std:: vector<double> > B = A;
	// std:: cout << other << std:: endl;


	for(int i = 0; i < A.size(); i++){
		for(int j = 0; j < A[0].size(); j++){
			// std:: cout << B[i][j] << " Before" << std:: endl;
			B[i][j] = other*A[i][j];
			// std:: cout << B[i][j] << " After" << std:: endl;
		}
	}
	Matrix C;
	C.mat_def(B);
	return C;
}

// Scalar multiplication 2
Matrix operator* (const double other, Matrix Z){
	std:: vector<std:: vector<double> > A = Z.mat_get();
	std:: vector<std:: vector<double> > B = A;
	// std:: cout << other << std:: endl;


	for(int i = 0; i < A.size(); i++){
		for(int j = 0; j < A[0].size(); j++){
			// std:: cout << B[i][j] << " Before" << std:: endl;
			B[i][j] = other*A[i][j];
			// std:: cout << B[i][j] << " After" << std:: endl;
		}
	}
	Matrix C;
	C.mat_def(B);
	// C.mat_print();

	return C;
}

// Scalar division
Matrix operator/ (Matrix Z, const double other){
	std:: vector<std:: vector<double> > A = Z.mat_get();
	std:: vector<std:: vector<double> > B = A;
	// std:: cout << other << std:: endl;


	for(int i = 0; i < A.size(); i++){
		for(int j = 0; j < A[0].size(); j++){
			// std:: cout << B[i][j] << " Before" << std:: endl;
			B[i][j] = A[i][j]/other;;
			// std:: cout << B[i][j] << " After" << std:: endl;
		}
	}
	Matrix C;
	C.mat_def(B);
	return C;
}

// Exponential
Matrix operator^ (Matrix Z, const int other){
	Matrix C;

	if(Z.N_rows() != Z.N_cols()){
		std:: cout << "Number of columns of the matrix_1 is not equal to number of rows matrix_2" << std:: endl;
		return C;
	}
	else{
		C = Z;
		for(int i = 1; i < other; i++){
			C = C*Z;
		}
	}

	return C;
}

// Matrix Addition
Matrix operator+ (Matrix Y, Matrix Z){
	std:: vector<std:: vector<double> > A = Y.mat_get();
	std:: vector<std:: vector<double> > B = Z.mat_get();
	std:: vector<std:: vector<double> > C = Z.mat_get();
	// Y.mat_print();
	// Z.mat_print();

	if((A.size() == B.size()) && (A[0].size() == B[0].size())){
		for(int i = 0; i < A.size(); i++){
			for(int j = 0; j < A[0].size(); j++){
				// std:: cout << A[i][j] << " ," << B[i][j] << " Before" << std:: endl;
				C[i][j] = A[i][j] + B[i][j];
				// std:: cout << C[i][j] << " After" << std:: endl;
			}
		}
	}
	else{
		std:: cout << "The matrices do not have same dimensions!" << std:: endl;
	}
	Matrix D;
	D.mat_def(C);
	return D;
}

// Matrix Subtraction
Matrix operator- (Matrix Y, Matrix Z){
	std:: vector<std:: vector<double> > A = Y.mat_get();
	std:: vector<std:: vector<double> > B = Z.mat_get();
	std:: vector<std:: vector<double> > C = Z.mat_get();
	// Y.mat_print();
	// Z.mat_print();

	if((A.size() == B.size()) && (A[0].size() == B[0].size())){
		for(int i = 0; i < A.size(); i++){
			for(int j = 0; j < A[0].size(); j++){
				// std:: cout << A[i][j] << " ," << B[i][j] << " Before" << std:: endl;
				C[i][j] = A[i][j] - B[i][j];
				// std:: cout << C[i][j] << " After" << std:: endl;
			}
		}
	}
	else{
		std:: cout << "The matrices do not have same dimensions!" << std:: endl;
	}
	Matrix D;
	D.mat_def(C);
	return D;
}

// Unary matrix negation
Matrix operator- (Matrix Z){
	std:: vector<std:: vector<double> > A = Z.mat_get();
	std:: vector<std:: vector<double> > B = Z.mat_get();
	// std:: vector<std:: vector<double>> C = Z.mat_get();
	// Y.mat_print();
	// Z.mat_print();

	for(int i = 0; i < A.size(); i++){
		for(int j = 0; j < A[0].size(); j++){
			// std:: cout << A[i][j] << " ," << B[i][j] << " Before" << std:: endl;
			B[i][j] = -A[i][j];
			// std:: cout << C[i][j] << " After" << std:: endl;
		}
	}

	Matrix C;
	C.mat_def(B);
	return C;
}

// Norm of matrix
double norm(Matrix _A){
	std:: vector<std:: vector<double> > A = _A.mat_get();

	int n = A.size();
	int m = A[0].size();

	double sum = 0;

	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			sum = sum + A[i][j]*A[i][j];
		}
	}

	sum = sqrt(sum);

	return sum;
}

// Identity matrix generator function
Matrix Identity(int n){
	std:: vector<std:: vector<double> > I;
	std:: vector<double> x;

	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(i != j){
				x.push_back(0);
			}
			else{
				x.push_back(1);
			}
		}
		I.push_back(x);
		x.clear();
	}

	Matrix _I;
	_I.mat_def(I);
	return _I;
}

// Zeros matrix generator function
Matrix Zeros(int n, int m){
	std:: vector<std:: vector<double> > Z;
	std:: vector<double> x;

	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			x.push_back(0);
		}
		Z.push_back(x);
		x.clear();
	}

	Matrix _Z;
	_Z.mat_def(Z);
	return _Z;
}

// Ones matrix generator function
Matrix Ones(int n, int m){
	std:: vector<std:: vector<double> > On;
	std:: vector<double> x;

	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			x.push_back(1);
		}
		On.push_back(x);
		x.clear();
	}

	Matrix _On;
	_On.mat_def(On);
	return _On;
}

// Solving a system of linear equations with Naive Gaussian elimination
Matrix naiveGaussianSolver(Matrix _A, Matrix _b){
	std:: vector<std:: vector<double> > A = _A.mat_get();
	std:: vector<std:: vector<double> > b = _b.mat_get();
	std:: vector<std:: vector<double> > x;

	int flag = 0;
	int n = A.size();

	if(A.size() != A[0].size()){
		std:: cout << "The _A matrix in Matrix naiveGaussianSolver(Matrix _A, Matrix _b) must have equal number of rows and columns!" << std:: endl;
	}
	else if(b[0].size() != 1){
		std:: cout << "The _b matrix in Matrix naiveGaussianSolver(Matrix _A, Matrix _b) must have only 1 element in each row!" << std:: endl;
	}
	else if(A[0].size() != b.size()){
		std:: cout << "In Matrix naiveGaussianSolver(Matrix _A, Matrix _b), the nummber of columns of _A matrix must be equal " << std:: endl;
		std:: cout << "to the number of rows of _b matrix!" << std:: endl;
	}
	else{
		flag = 1;
	}

	if(flag == 1){
		// Forward elimination with partial pivoting
		std:: vector<std:: vector<int> > pivoting;
		std:: vector<int> t;
		double maximum;

		// pivoting = {{0,0}};
		/*
		for(int i = 0; i < n; i++){
			maximum = A[i][i];
			for(int j = i+1; j < n; j++){
				if(A[j][i] == 0){
					continue;
				}
				else{
					if((maximum/A[j][i] < 1) && (maximum/A[j][i] > -1)){
						t.clear();
						t.push_back(i);
						t.push_back(j);
						maximum = A[j][i];
					}
				}
			}
			if(t.empty() != true){
				pivoting.push_back(t);
				t.clear();
			}
		}
		 */

		double temp;
		double perm;

		for(int i = 0; i < n; i++){
			maximum = A[i][i];
			for(int j = i+1; j < n; j++){
				if(A[j][i] == 0){
					continue;
				}
				else{
					if((maximum/A[j][i] < 1) && (maximum/A[j][i] > -1)){
						t.clear();
						t.push_back(i);
						t.push_back(j);
						maximum = A[j][i];
					}
				}
			}
			if(t.empty() == false){
				for(int k = 0; k < n; k++){
					temp = A[i][k];
					A[i][k] = A[t[1]][k];
					A[t[1]][k] = temp;
				}

				temp = b[i][0];
				b[i][0] = b[t[1]][0];
				b[t[1]][0] = temp;

				pivoting.push_back(t);
				t.clear();
			}

			/*
			Matrix __A;
			__A.mat_def(A);
			text_print(A* =);
			__A.mat_print();

			Matrix __b;
			__b.mat_def(b);
			text_print(b* =);
			__b.mat_print();
			 */

			for(int j = i+1; j < n; j++){

				perm = A[j][i];
				temp = b[j][0] - (perm/A[i][i])*b[i][0];
				b[j][0] = temp;
				if((b[j][0] <= 1e-16) && (b[j][0] >= -1e-16)){
					b[j][0] = 0;
				}

				for(int k = i; k < n; k++){

					temp = A[j][k] - (perm/A[i][i])*A[i][k];
					A[j][k] = temp;

					if((A[j][k] <= 1e-16) && (A[j][k] >= -1e-16)){
						A[j][k] = 0;
					}

				}
			}
		}

		/*
		Matrix<int> _pivoting;
		_pivoting.mat_def(pivoting);
		text_print(Pivoting =);
		_pivoting.mat_print();
		 */

		// Back substitution
		std:: vector<double> TEMP;
		double sum;

		for(int i = n-1; i >= 0; i--){
			sum = b[i][0];
			for(int j = n-1; j > i; j--){
				temp = sum - A[i][j]*x[n-1-j][0];
				sum = temp;
			}
			temp = sum/A[i][i];
			sum = temp;

			if((sum <= 1e-16) && (sum >= -1e-16)){
				sum = 0;
			}

			TEMP.push_back(sum);
			x.push_back(TEMP);
			TEMP.clear();
		}
	}

	double temp;

	for(int i = 0; i < n/2; i++){
		temp = x[i][0];
		x[i][0] = x[n-1-i][0];
		x[n-1-i][0] = temp;
	}

	Matrix _x;
	_x.mat_def(x);
	return _x;
}

// Solving a system of linear equations with Gauss-Seidel method
Matrix GaussSeidel_SOR_Solver(Matrix _A, Matrix _b, double w, double tol){
	std:: vector<std:: vector<double> > A = _A.mat_get();
	std:: vector<std:: vector<double> > b = _b.mat_get();
	std:: vector<std:: vector<double> > x;
	Matrix _x;

	int flag = 0;
	int n = A.size();

	if(A.size() != A[0].size()){
		std:: cout << "The _A matrix in Matrix GaussSeidel_SOR_Solver(Matrix _A, Matrix _b, double w, double tol) must have equal number of rows and columns!" << std:: endl;
	}
	else if(b[0].size() != 1){
		std:: cout << "The _b matrix in Matrix GaussSeidel_SOR_Solver(Matrix _A, Matrix _b, double w, double tol) must have only 1 element in each row!" << std:: endl;
	}
	else if(A[0].size() != b.size()){
		std:: cout << "In Matrix GaussSeidel_SOR_Solver(Matrix _A, Matrix _b, double w, double tol), the nummber of columns of _A matrix must be equal " << std:: endl;
		std:: cout << "to the number of rows of _b matrix!" << std:: endl;
	}
	else{
		flag = 1;
	}

	if(flag == 1){
		// Initial assignment of x vector
		/*
		std:: vector<double> TEMP;
		TEMP = {0};
		for(int i = 0; i < n; i++){
			x.push_back(TEMP);
		}
		*/
		_x = Zeros(n,1);

		// Iterations
		double r;
		double correction;
		int itr = 1;

		while(norm(_A*_x - _b) > tol){

			if(itr == 501){
				break;
			}

			// text_print(_x = )
			// _x.mat_print();

			for(int i = 0; i < n; i++){
				r = b[i][0];
				for(int j = 0; j < n; j++){
					r = r - A[i][j]*x[j][0];
				}
				correction = r/A[i][i];
				x[i][0] = x[i][0] + w*correction;
			}

			itr++;
			_x.mat_def(x);
		}
		std:: cout << "Gauss-Seidel method with SOR:" << std:: endl;
		std:: cout << "Relaxation factor = " << w << std:: endl;
		std:: cout << "Tolerance = " << tol << std:: endl;
		std:: cout << "Number of iterations = " << itr << "\n";
		std:: cout << "\n";
	}


	return _x;
}

// Finding the dominant eigenvector from Power method
Matrix PowerMethod(Matrix _A, double tol){
	// std:: vector<std:: vector<double>> v;
	int n = _A.N_rows();
	int m = _A.N_cols();

	Matrix _v = Zeros(m,1);
	_v.element(0, 0, 1);

	/*
	int first = 1;

	for(int i = 0; i < m; i++){
		if(first == 1){
			v = {{1}};
			first = 0;
		}
		else{
			v.push_back({0});
		}
	}

	_v.mat_def(v);
	*/
	int flag = 0;

	if(n != m){
		std:: cout << "The _A matrix in Matrix PowerMethod(Matrix _A, double tol) must have equal number of rows and columns!" << std:: endl;
	}
	else{
		flag = 1;
	}

	if(flag == 1){
		Matrix _previous;
		int first = 1;
		int itr;

		// Iterations
		for(itr = 1; itr < 501; itr++){
			_v = _A*_v/norm(_A*_v);
			// v = _v.mat_get();

			if(_v.element(0,0) < 0){ // Keeping the first element of the vector positive
				_v = -_v;
			}

			if(first == 1){
				first = 0;
				_previous = _v;
				continue;
			}
			else{
				if(norm(_v-_previous) <= tol){
					break;
				}
				else{
					_previous = _v;
					continue;
				}
			}
		}
		std:: cout << "Power Method:" << "\n";
		std:: cout << "Tolerance = " << tol << "\n";
		std:: cout << "Number of iterations = " << itr << "\n\n";
	}

	return _v;
}

// Finding the other eigenvectors from Inverse Power method
Matrix InversePowerMethod(Matrix _A, double shift, double tol){
	//std:: vector<std:: vector<double>> v;
	int n = _A.N_rows();
	int m = _A.N_cols();

	Matrix _v = Zeros(m,1);
	_v.element(0, 0, 1);

	/*
	int first = 1;

	for(int i = 0; i < m; i++){
		if(first == 1){
			v = {{1}};
			first = 0;
		}
		else{
			v.push_back({0});
		}
	}

	_v.mat_def(v);
	*/

	int flag = 0;

	if(n != m){
		std:: cout << "The _A matrix in Matrix PowerMethod(Matrix _A, double tol) must have equal number of rows and columns!" << std:: endl;
	}
	else{
		flag = 1;
	}

	if(flag == 1){
		Matrix _previous;
		Matrix _new_A;

		_new_A = _A - shift*Identity(n);

		int first = 1;
		int itr;

		// Iterations
		for(itr = 1; itr < 501; itr++){
			_v = naiveGaussianSolver(_new_A,_v);
			// text_print(_v =);
			// _v.mat_print();
			_v = _v/norm(_v);
			// v = _v.mat_get();

			if(_v.element(0,0) < 0){ // Keeping the first element of the vector positive
				_v = -_v;
			}


			if(first == 1){
				first = 0;
				_previous = _v;
				continue;
			}
			else{
				if(norm(_v-_previous) <= tol){
					break;
				}
				else{
					/*
					text_print(_previous =);
					_previous.mat_print();
					text_print(_v =);
					_v.mat_print();
					std:: cout << "NORM = " << norm(_v-_previous) << "\n\n";
					 */
					_previous = _v;
					continue;
				}
			}
		}
		std:: cout << "Inverse Power Method:" << "\n";
		std:: cout << "Shift = " << shift << "\n";
		std:: cout << "Tolerance = " << tol << "\n";
		std:: cout << "Number of iterations = " << itr << "\n\n";
	}

	return _v;
}


} /* namespace mat */
