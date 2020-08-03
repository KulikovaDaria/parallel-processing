#include "vector.h"
#include "parallel_vector.h"
#include "matrix.h"
#include "parallel_matrix.h"

#include <iostream>
#include <omp.h>

using namespace std;

Vector GaussianElimination(Matrix A, Vector b) {
  const int n = b.Size();
  Matrix T(n);
  for (int k = 0; k < n - 1; ++k) {
    for (int i = k + 1; i < n; ++i) {
      T(i, k) = A(i, k) / A(k, k);
      b[i] -= T(i, k) * b[k];
      for (int j = k + 1; j < n; ++j) {
        A(i, j) -= T(i, k) * A(k, j);
      }
    }
  }
  Vector x(n);
  x[n - 1] = b[n - 1] / A(n - 1, n - 1);
  for (int k = n - 2; k >= 0; --k) {
    double s = 0;
    for (int j = k + 1; j < n; ++j) {
      s += A(k, j) * x[j];
    }
    x[k] = (b[k] - s) / A(k, k);
  }
  return x;
}



ParallelVector GaussianElimination(ParallelMatrix A, ParallelVector b) {
  const int n = b.Size();
  omp_set_num_threads(100);
  int num = -1;
  ParallelMatrix T(n);
  for (int k = 0; k < n - 1; ++k) {
#pragma omp parallel for num_threads(99)
    for (int i = 0; i < n; ++i) {
      T(i, k) = A(i, k) / A(k, k);
      b[i] = T(i, k) * b[k];
#pragma omp parallel for num_threads(99)
      for (int j = 0; j < n; ++j) {
        A(i, j) = T(i, k) * A(k, j);  
        A(i, j) *= k;
        num = omp_get_num_threads();
      }
    }
  }
  cout << num << "    KKKKKKKKK     " << endl;
  ParallelVector x(n);
  x[n - 1] = b[n - 1] / A(n - 1, n - 1);
  for (int k = n - 2; k >= 0; --k) {
    double s = 0;
#pragma omp parallel for reduction(+:s) num_threads(n - k - 1)
    for (int j = k + 1; j < n; ++j) {
      s += A(k, j) * x[j];
    }
    x[k] = (b[k] - s) / A(k, k);
  }
  return x;
}