#include <iostream>

#include <vector>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

#include "redsvd.hpp"

using namespace std;
using namespace Eigen;
using namespace REDSVD;

void createMatrixA(const double* dataValuesRowMajor, int numberOfRows, int numberOfColumns, MatrixXd& A)
{
	A.resize(numberOfRows, numberOfColumns);
	for (int i = 0; i < numberOfRows; i++)
	{
		for (int j = 0; j < numberOfColumns; j++)
		{
			A(i, j) = dataValuesRowMajor[i * numberOfColumns + j];
		}
	}
}

template <class Svd>
void assembleOutput(Svd& svd, /*out*/ double* singularValues, /*out*/ double* U_VT_ColumnMajor)
{
	const MatrixXd& U = svd.matrixU();
	const VectorXd& S = svd.singularValues();
	const MatrixXd& V = svd.matrixV();

	// S
	for (int i = 0; i < S.rows(); i++)
	{
		singularValues[i] = S(i);
	}

	// U_VT_ColumnMajor length should be: U.rows() * U.cols() + V.cols() * V.rows());

	int index = 0;
	// U
	for (int j = 0; j < U.cols(); j++)
	{
		for (int i = 0; i < U.rows(); i++)
		{
			U_VT_ColumnMajor[index] = U(i, j);
			index++;
		}
	}

	// V transposed
	for (int i = 0; i < V.rows(); i++)
	{
		for (int j = 0; j < V.cols(); j++)
		{
			U_VT_ColumnMajor[index] = V(i, j);
			index++;
		}
	}
}

extern "C"
{
	__declspec(dllexport) void __stdcall ComputeSvdExact(const double* dataValuesRowMajor, int numberOfRows, int numberOfColumns, /*out*/ double* singularValues, /*out*/ double* U_VT_ColumnMajor)
	{
		MatrixXd A;
		createMatrixA(dataValuesRowMajor, numberOfRows, numberOfColumns, A);
		Eigen::JacobiSVD<Eigen::MatrixXd/*, Eigen::NoQRPreconditioner*/> svd_exact(A, Eigen::ComputeThinU | Eigen::ComputeThinV); // compute
		assembleOutput(svd_exact, singularValues, U_VT_ColumnMajor);
	}

	__declspec(dllexport) void __stdcall ComputeSvdRandomized(const double* dataValuesRowMajor, int numberOfRows, int numberOfColumns, int rank, /*out*/ double* singularValues, /*out*/ double* U_VT_ColumnMajor)
	{
		MatrixXd A;
		createMatrixA(dataValuesRowMajor, numberOfRows, numberOfColumns, A);
		RedSVD svd_approx(A, rank); // compute
		assembleOutput(svd_approx, singularValues, U_VT_ColumnMajor);
	}
}