#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>


std::vector<double> GaussianElimination(const std::vector<std::vector<double>>& matr, const std::vector<double>& vec) {
  const int n = vec.size();
  std::vector<std::vector<double>> A(n, std::vector<double>(n));
  std::vector<double> b(n);
  for (int i = 0; i < n; ++i) {
    b[i] = vec[i];
    for (int j = 0; j < n; ++j) {
      A[i][j] = matr[i][j];
    }
  }
  for (int k = 0; k < n - 1; ++k) {
    for (int i = k + 1; i < n; ++i) {
      double t = A[i][k] / A[k][k];
      b[i] -= t * b[k];
      for (int j = k + 1; j < n; ++j) {
        A[i][j] -= t * A[k][j];
      }
    }
  }
  std::vector<double> x(n);
  x[n - 1] = b[n - 1] / A[n - 1][n - 1];
  for (int k = n - 2; k >= 0; --k) {
    double s = 0;
    for (int j = k + 1; j < n; ++j) {
      s += A[k][j] * x[j];
    }
    x[k] = (b[k] - s) / A[k][k];
  }
  return x;
}

std::vector<double> ParallelGaussianElimination(const std::vector<std::vector<double>>& matr, const std::vector<double>& vec) {
  //omp_set_num_threads(8);
  const int n = vec.size();
  std::vector<std::vector<double>> A(n, std::vector<double>(n));
  std::vector<double> b(n);
#pragma omp parallel for
  for (int ij = 0; ij < n * n; ++ij) {
    int i = ij / n;
    int j = ij % n;
    b[i] = vec[i];
    A[i][j] = matr[i][j];
  }
  for (int k = 0; k < n - 1; ++k) {
#pragma omp parallel for if(k < 800)
    for (int i = k + 1; i < n; ++i) {
      double t = A[i][k] / A[k][k];
      b[i] -= t * b[k];
      for (int j = k + 1; j < n; ++j) {
        A[i][j] -= t * A[k][j];
      }
    }
  }
  std::vector<double> x(n);
  x[n - 1] = b[n - 1] / A[n - 1][n - 1];
  for (int k = n - 2; k >= 0; --k) {
    double s = 0;
//#pragma omp parallel for reduction(+:s)
    for (int j = k + 1; j < n; ++j) {
      s += A[k][j] * x[j];
    }
    x[k] = (b[k] - s) / A[k][k];
  }

  return x;
}

std::vector<double> JacobiMethod(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
  int n = b.size();
  std::vector<double> x(n, 0), next_x(n);
  double norm = 1;
  double eps = 1e-3;
  while (norm > eps) {
    norm = -1;
    for (int i = 0; i < n; ++i) {
      next_x[i] = b[i];
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          next_x[i] -= A[i][j] * x[j];
        }
      }
      next_x[i] /= A[i][i];
      norm = std::max(fabs(x[i] - next_x[i]), norm);
    }
    swap(x, next_x);
  }
  return x;
}

std::vector<double> ParallelJacobiMethod(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
  int n = b.size();
  std::vector<double> x(n, 0), next_x(n);
  double norm = 1;
  double eps = 1e-3;
  while (norm > eps) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      next_x[i] = b[i];
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          next_x[i] -= A[i][j] * x[j];
        }
      }
      next_x[i] /= A[i][i];
    }
    norm = fabs(x[0] - next_x[0]);
    for (int i = 1; i < n; ++i) {
      norm = std::max(fabs(x[i] - next_x[i]), norm);
    }
    swap(x, next_x);
  }
  return x;
}

std::vector<double> SeidelMethod(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
  int n = b.size();
  std::vector<double> x(n, 0);
  double norm = 1;
  double eps = 1e-3;
  while (norm > eps) {
    norm = -1;
    for (int i = 0; i < n; ++i) {
      double next_x = b[i];
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          next_x -= A[i][j] * x[j];
        }
      }
      next_x /= A[i][i];
      norm = std::max(fabs(x[i] - next_x), norm);
      x[i] = next_x;
    }
  }
  return x;
}

std::vector<double> ParallelSeidelMethod(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
  int n = b.size();
  std::vector<double> x(n, 0);
  double norm = 1;
  std::vector<double> vec_norm(n);
  double eps = 1e-3;
  while (norm > eps) {
    norm = -1;
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      double next_x = b[i];
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          next_x -= A[i][j] * x[j];
        }
      }
      next_x /= A[i][i];
      vec_norm[i] = fabs(x[i] - next_x);
      x[i] = next_x;
    }
    for (int i = 0; i < n; ++i) {
      norm = std::max(vec_norm[i], norm);
    }
  }
  return x;
}

std::vector<double> MonteCarloMethod(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
  int n = b.size();
  std::vector<std::vector<double>> B(A);
  std::vector<double> f(n, 0);
  std::vector<double> row_sum(n, 0);

  double ma = 1;
  for (int i = 0; i < n; ++i) {
    ma = std::max(ma, fabs(A[i][i]));
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == j) {
        B[i][j] = (1 - A[i][j] / ma);
      }
      else {
        B[i][j] = -A[i][j] / ma;
      }
      row_sum[i] += fabs(B[i][j]);
    }
    f[i] = b[i] / ma;
  }

  std::vector<std::vector<double>> P(n, std::vector<double>(n, 0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      P[i][j] = fabs(B[i][j]) / row_sum[i];
    }
  }

  std::vector<double> s(f);
  std::vector<double> s_sum(f);
  int m = 100000;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double p = (double)(rand()) / RAND_MAX;
      int k = 0;
      for (double su = P[j][0]; k < n; ++k, su += P[j][k]) {
        if (su >= p - 1e-5) {
          break;
        }
      }
      s[j] = f[j] + s[k] * B[j][k] / P[j][k];
      s_sum[j] += s[j];
    }
  }
  for (int i = 0; i < n; ++i) {
    s[i] = s_sum[i] / (m + 1);
  }
  return s;
}

std::vector<double> ParallelMonteCarloMethod(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
  int n = b.size();
  std::vector<std::vector<double>> B(A);
  std::vector<double> f(n, 0);
  std::vector<double> row_sum(n, 0);

  double ma = 1;
  for (int i = 0; i < n; ++i) {
    ma = std::max(ma, fabs(A[i][i]));
  }
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == j) {
        B[i][j] = (1 - A[i][j] / ma);
      }
      else {
        B[i][j] = -A[i][j] / ma;
      }
      row_sum[i] += fabs(B[i][j]);
    }
    f[i] = b[i] / ma;
  }

  std::vector<std::vector<double>> P(n, std::vector<double>(n, 0));
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      P[i][j] = fabs(B[i][j]) / row_sum[i];
    }
  }

  std::vector<double> s(f);
  std::vector<double> s_sum(f);
  int m = 100000;
  for (int i = 0; i < m; ++i) {
#pragma omp parallel for
    for (int j = 0; j < n; ++j) {
      double p = (double)(rand()) / RAND_MAX;
      int k = 0;
      for (double su = P[j][0]; k < n; ++k, su += P[j][k]) {
        if (su >= p - 1e-5) {
          break;
        }
      }
      s[j] = f[j] + s[k] * B[j][k] / P[j][k];
      s_sum[j] += s[j];
    }
  }
  for (int i = 0; i < n; ++i) {
    s[i] = s_sum[i] / (m + 1);
  }
  return s;
}

void RandVector(std::vector<double>& vec) {
  for (int i = 0; i < vec.size(); ++i) {
    vec[i] = rand();
  }
}


void RandMatrix(std::vector<std::vector<double>>& matr) {
  for (int i = 0; i < matr.size(); ++i) {
    long long sum = 0;
    for (int j = 0; j < matr[i].size(); ++j) {
      matr[i][j] = rand();
      sum += matr[i][j];
    }
    sum -= matr[i][i];
    if (sum >= 0) {
      matr[i][i] = sum + abs(matr[i][i]) + 1;
    }
    else {
      matr[i][i] = sum - abs(matr[i][i]) - 1;
    }
  }
}

int main() {
  std::cout << std::setprecision(3);

  int n = 3;
  std::vector<std::vector<double>> A{{0.7, 0.5, -0.1}, {0.2, 0.7, -0.4}, {-0.4, 0.3, 0.8}};
  std::vector<double> b{0.1, -0.5, 0.4};
  std::cout << "A = " << std::endl;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << A[i][j] << '   ';
    }
    std::cout << std::endl;
  }
  std::cout << "b = " << std::endl;
  for (int i = 0; i < n; ++i) {
    std::cout << b[i] << '   ';
  }
  std::cout << std::endl << std::endl;

  std::vector<double> x = MonteCarloMethod(A, b);
  std::cout << "Monte Carlo Method:   x = ";
  for (int i = 0; i < b.size(); ++i) {
    std::cout << x[i] << ' ';
  }
  std::cout << std::endl;
  x = ParallelMonteCarloMethod(A, b);
  std::cout << "Parallel Monte Carlo Method:   x = ";
  for (int i = 0; i < b.size(); ++i) {
    std::cout << x[i] << ' ';
  }
  std::cout << std::endl;
  x = SeidelMethod(A, b);
  std::cout << "Seidel Method:   x = ";
  for (int i = 0; i < b.size(); ++i) {
    std::cout << x[i] << ' ';
  }
  std::cout << std::endl << std::endl << std::endl;

  n = 100;
  std::vector<std::vector<double>> A1(n, std::vector<double>(n));  RandMatrix(A1);
  std::vector<double> b1(n); RandVector(b1);
  /*std::cout << "A = " << std::endl;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << A1[i][j] << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << "b = " << std::endl;
  for (int i = 0; i < n; ++i) {
    std::cout << b1[i] << ' ';
  }
  std::cout << std::endl << std::endl;*/

 // std::vector<double> x1 = MonteCarloMethod(A1, b1);
  std::vector<double> x2 = ParallelMonteCarloMethod(A1, b1);
  std::vector<double> x3 = SeidelMethod(A1, b1);

  double eps = 0;
  for (int i = 0; i < n; ++i) {
    eps += fabs(x2[i] - x3[i]);
  }
  std::cout << "Parallel Monte Carlo Method:   eps = " << eps << std::endl;





  n = 1000;
  int q = 10;
  std::vector<std::vector<double>> seq(3), par(3);

  for (int i = 0; i < q; ++i) {
    std::vector<double> vec(n); RandVector(vec);
    std::vector<std::vector<double>> matr(n, std::vector<double>(n, 0)); RandMatrix(matr);
    std::vector<double> x;
    double start, finish;

    start = clock();
    x = GaussianElimination(matr, vec);
    finish = clock();
    seq[0].push_back(finish - start);

    start = clock();
    x = ParallelGaussianElimination(matr, vec);
    finish = clock();
    par[0].push_back(finish - start);

    start = clock();
    x = JacobiMethod(matr, vec);
    finish = clock();
    seq[1].push_back(finish - start);

    start = clock();
    x = ParallelJacobiMethod(matr, vec);
    finish = clock();
    par[1].push_back(finish - start);

    start = clock();
    x = SeidelMethod(matr, vec);
    finish = clock();
    seq[2].push_back(finish - start);

    start = clock();
    x = ParallelSeidelMethod(matr, vec);
    finish = clock();
    par[2].push_back(finish - start);
  }

  std::vector<double> s(3), p(3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < q; ++j) {
      s[i] += seq[i][j];
      p[i] += par[i][j];
    }
  }
  
  std::cout << "Gaussian Elimination's coefficient: " << s[0] / p[0] << std::endl;
  std::cout << "Jacobi Method's coefficient: " << s[1] / p[1] << std::endl;
  std::cout << "Seidel Method's coefficient: " << s[2] / p[2] << std::endl;
  

   return 0;
}