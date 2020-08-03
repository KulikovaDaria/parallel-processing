#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "vector.h"
#include "matrix.h"
#include "parallel_vector.h"
#include "parallel_matrix.h"

#include <vector>
#include <omp.h>

void RandVector(Vector& vec) {
  for (int i = 0; i < vec.Size(); ++i) {
    vec[i] = rand();
  }
}

void RandVector(ParallelVector& vec) {
  for (int i = 0; i < vec.Size(); ++i) {
    vec[i] = rand();
  }
}

void RandMatrix(Matrix& matr) {
  for (int i = 0; i < matr.RowNum(); ++i) {
    for (int j = 0; j < matr.ColumnNum(); ++j) {
      matr(i, j) = -10000 + abs(rand()) % 20000;
      matr(i, j) /= 10000;
    }
  }
}

void RandMatrix(ParallelMatrix& matr) {
  for (int i = 0; i < matr.RowNum(); ++i) {
    for (int j = 0; j < matr.ColumnNum(); ++j) {
      matr(i, j) = rand();
    }
  }
}

int main() {

  std::cout << std::setprecision(3);

/*  int vsize = 1e7;
  Vector v1(vsize), v2(vsize);
  RandVector(v1); RandVector(v2);
  ParallelVector pv1(vsize), pv2(vsize);
  RandVector(pv1); RandVector(pv2);

  std::cout << "Vector's addition" << std::endl;
  Vector v3 = v1 + v2;
  ParallelVector pv3 = pv1 + pv2;
  std::cout << std::endl;
  std::cout << "Vector's subtraction" << std::endl;
  v3 = v1 - v2;
  pv3 = pv1 - pv2;
  std::cout << std::endl;
  std::cout << "Multiplying a vector by a scalar" << std::endl;
  double a = rand();
  v3 = a * v1;
  pv3 = a * pv1;
  std::cout << std::endl;
  std::cout << "Vector's length" << std::endl;
  v1.Length();
  pv1.Length();
  std::cout << std::endl;
  std::cout << "Scalar product" << std::endl;
  double sp = v1 * v2;
  sp = pv1 * pv2;
  std::cout << std::endl;

  int msize = 1e3;
  Matrix m1(msize), m2(msize);
  RandMatrix(m1); RandMatrix(m2);
  ParallelMatrix pm1(msize), pm2(msize);
  RandMatrix(pm1); RandMatrix(pm2);
  std::cout << "Matrix's addition" << std::endl;
  Matrix m3 = m1 + m2;
  ParallelMatrix pm3 = pm1 + pm2;
  std::cout << std::endl;
  std::cout << "Matrix's subtraction" << std::endl;
  m3 = m1 - m2;
  pm3 = pm1 - pm2;
  std::cout << std::endl;
  std::cout << "Multiplying a matrix by a scalar" << std::endl;
  a = rand();
  m3 = a * m1;
  pm3 = a * pm1;
  std::cout << std::endl;
  std::cout << "Matrix's multiplication" << std::endl;
  m3 = m1 * m2;
  pm3 = pm1 * pm2;
  std::cout << std::endl;
  std::cout << "Multiplying a matrix by a vector" << std::endl;
  Vector v(msize); RandVector(v);
  ParallelVector pv(msize); RandVector(pv);
  v = m1 * v;
  pv = pm1 * pv;
  std::cout << std::endl;
  std::cout << "Matrix's transpose" << std::endl;
  m3 = m1.Transpose();
  pm3 = pm1.Transpose();
  std::cout << std::endl;
  std::cout << "Frobenius Norm" << std::endl;
  double norm = m1.FrobeniusNorm();
  norm = pm1.FrobeniusNorm();
  std::cout << std::endl; */


  /*std::cout << "----------DETERMINANT----------" << std::endl;
  Matrix m1{{3, 4, 2}, {2, -1, -3}, {1, 5, 1}};
  ParallelMatrix m2(m1.RowNum());
  for (int i = 0; i < m1.RowNum(); ++i) {
    for (int j = 0; j < m1.ColumnNum(); ++j) {
      m2(i, j) = m1(i, j);
    }
  }
  std::cout << m1 << std::endl;
  std::cout << "Sequential determinant: " << m1.Determinant() << std::endl;
  std::cout << "Parallel determinant: " << m2.Determinant() << std::endl;
  int n = 1000;
  int q = 1;
  double cf1 = 0, cf2 = 0;
  for (int i = 0; i < q; ++i) {
    Matrix m1(n); RandMatrix(m1);
    double start = clock();
    double det1 = m1.Determinant();
    double finish = clock();
    cf1 += finish - start;

    ParallelMatrix m2(n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        m2(i, j) = m1(i, j);
      }
    }
    start = clock();
    double det2 =  m2.Determinant();
    finish = clock();
    cf2 += finish - start;
  }
  std::cout << "Coefficient = " << cf1 / cf2 << std::endl;
  std::cout << std::endl;

  std::cout << "----------INVERTIBLE MATRIX----------" << std::endl;
  std::cout << m1 << std::endl;
  std::cout << "Sequential invertible matrix: " << m1.Invertible() << std::endl;
  std::cout << "Parallel invertible matrix: " << m2.Invertible() << std::endl;
  std::cout << "Matrix * InvertableMatrix = " << m1 * m1.Invertible() << std::endl;
  cf1 = 0, cf2 = 0;
  for (int i = 0; i < q; ++i) {
    Matrix m1(n); RandMatrix(m1);
    double start = clock();
    Matrix im1 = m1.Invertible();
    double finish = clock();
    cf1 += finish - start;

    ParallelMatrix m2(n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        m2(i, j) = m1(i, j);
      }
    }
    start = clock();
    ParallelMatrix im2 = m2.Invertible();
    finish = clock();
    cf2 += finish - start;
  }
  std::cout << "Coefficient = " << cf1 / cf2 << std::endl;
  std::cout << std::endl;

  std::cout << "----------LU DECOMPOSITION----------" << std::endl;
  Matrix m3{{2, -1, 1}, {4, 3, 1}, {6, -13, 6}};
  m1 = m3;
  for (int i = 0; i < m1.RowNum(); ++i) {
    for (int j = 0; j < m1.ColumnNum(); ++j) {
      m2(i, j) = m1(i, j);
    }
  }
  std::cout << m1 << std::endl;
  Matrix L1(m1.RowNum()), U1(m1.RowNum());
  ParallelMatrix L2(m1.RowNum()), U2(m1.RowNum());
  for (int i = 0; i < m1.RowNum(); ++i) {
    L1(i, i) = 1;
    L2(i, i) = 1;
  }
  m1.LUdecomposition(L1, U1);
  m2.LUdecomposition(L2, U2);
  std::cout << "Sequential L: " << L1 << std::endl;
  std::cout << "Parallel L: " << L2 << std::endl;
  std::cout << "Sequential U: " << U1 << std::endl;
  std::cout << "Parallel U: " << U2 << std::endl;
  cf1 = 0, cf2 = 0;
  for (int i = 0; i < q; ++i) {
    Matrix m1(n); RandMatrix(m1);
    Matrix L1(n), U1(n);
    double start = clock();
    m1.LUdecomposition(L1, U1);
    double finish = clock();
    cf1 += finish - start;

    ParallelMatrix m2(n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        m2(i, j) = m1(i, j);
      }
    }
    ParallelMatrix L2(n), U2(n);
    start = clock();
    m2.LUdecomposition(L2, U2);
    finish = clock();
    cf2 += finish - start;
  }
  std::cout << "Coefficient = " << cf1 / cf2 << std::endl;
  std::cout << std::endl; */

  std::cout << "----------SPECTRAL NORM----------" << std::endl;
  Matrix m1{{7, 2, -5}, {-9, 8, -5}, {24, -6, 8}};//{{1, -1}, {-1, 2}};
  ParallelMatrix m2(m1.RowNum());
  for (int i = 0; i < m1.RowNum(); ++i) {
    for (int j = 0; j < m1.ColumnNum(); ++j) {
      m2(i, j) = m1(i, j);
    }
  }
  std::cout << m1 << std::endl;
  std::cout << "Sequential norm: " << m1.SpectralNorm() << std::endl;
  std::cout << "Parallel norm: " << m2.SpectralNorm() << std::endl;
  int n = 1000;
  int q = 1;
  double cf1 = 0, cf2 = 0;
  for (int i = 0; i < q; ++i) {
    Matrix m1(n); RandMatrix(m1);
    double start = clock();
    double det1 = m1.SpectralNorm();
    double finish = clock();
    cf1 += finish - start;

    ParallelMatrix m2(n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        m2(i, j) = m1(i, j);
      }
    }
    start = clock();
    double det2 = m2.SpectralNorm();
    finish = clock();
    cf2 += finish - start;
  }
  std::cout << "Coefficient = " << cf1 / cf2 << std::endl;
  std::cout << std::endl;


  return 0;
}