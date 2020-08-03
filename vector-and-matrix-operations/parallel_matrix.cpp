#include "parallel_matrix.h"
#include <ctime>
#include <exception>
#include <math.h>
#include <omp.h>
#include <sstream>

ParallelMatrix::ParallelMatrix(const ptrdiff_t size)
  :ParallelMatrix(size, size) {}



ParallelMatrix::ParallelMatrix(const ptrdiff_t i_size, const ptrdiff_t j_size)
  : i_size_(i_size), j_size_(j_size), i_capacity_(i_size),
  j_capacity_(j_size) {
  if (i_size < 0 || j_size < 0) {
    throw std::length_error("Size cant't be < 0");
  }
  data_ = new double[i_size * j_size];
#pragma omp parallel for
  for (ptrdiff_t i = 0; i < i_size * j_size; ++i) {
    *(data_ + i) = 0;
  }
}



ParallelMatrix::ParallelMatrix(const ParallelMatrix& obj)
  : i_size_(obj.i_size_), j_size_(obj.j_size_), i_capacity_(obj.i_size_),
  j_capacity_(obj.j_size_), data_(new double[obj.i_size_ * obj.j_size_]) {
  Copy(obj, data_, j_capacity_);
}



ParallelMatrix::ParallelMatrix(const std::initializer_list<std::initializer_list<double>>& data)
  :ParallelMatrix(data.size(), data.begin()->size()) {
  int i = 0, j = 0;
  int size = data.begin()->size();
  for (auto row : data) {
    if (row.size() != size) {
      throw std::length_error("Rows size must be equal");
    }
    for (double element : row) {
      (*this)(i, j++) = element;
    }
    ++i;
    j = 0;
  }
}



ParallelMatrix::~ParallelMatrix() noexcept {
  delete[] data_;
  data_ = nullptr;
}



ParallelMatrix& ParallelMatrix::operator=(const ParallelMatrix& obj) {
  if (this != &obj) {
    if (i_capacity_ < obj.i_size_ || j_capacity_ < obj.j_size_) {
      Resize(obj.i_size_, obj.j_size_);
    }
    Copy(obj, data_, j_capacity_);
    i_size_ = obj.i_size_;
    j_size_ = obj.j_size_;
  }
  return *this;
}



double& ParallelMatrix::operator()(const ptrdiff_t i, const ptrdiff_t j) {
  return *(data_ + i * j_capacity_ + j);
}



double ParallelMatrix::operator()(const ptrdiff_t i, const ptrdiff_t j) const {
  return *(data_ + i * j_capacity_ + j);
}



ParallelMatrix& ParallelMatrix::operator+=(const ParallelMatrix& obj) {
  if (i_size_ != obj.i_size_ || j_size_ != obj.j_size_) {
    throw std::length_error("Matrixes must be the equal size");
  }
#pragma omp parallel for
  for (int i = 0; i < i_size_; ++i) {
#pragma omp parallel for
    for (int j = 0; j < j_size_; ++j) {
      (*this)(i, j) += obj(i, j);
    }
  }
  return *this;
}



ParallelMatrix& ParallelMatrix::operator-=(const ParallelMatrix& obj) {
  if (i_size_ != obj.i_size_ || j_size_ != obj.j_size_) {
    throw std::length_error("Matrixes must be the equal size");
  }
#pragma omp parallel for
  for (int i = 0; i < i_size_; ++i) {
#pragma omp parallel for
    for (int j = 0; j < j_size_; ++j) {
      (*this)(i, j) -= obj(i, j);
    }
  }
  return *this;
}



ParallelMatrix& ParallelMatrix::operator*=(const double a) {
#pragma omp parallel for
  for (int i = 0; i < i_size_; ++i) {
#pragma omp parallel for
    for (int j = 0; j < j_size_; ++j) {
      (*this)(i, j) *= a;
    }
  }
  return *this;
}



ParallelMatrix& ParallelMatrix::operator*=(const ParallelMatrix& obj) {
  *this = *this * obj;
  return *this;
}



ParallelMatrix ParallelMatrix::Transpose() const {
  double start = clock();
  ParallelMatrix new_matrix(j_size_, i_size_);
#pragma omp parallel for
  for (int i = 0; i < i_size_; ++i) {
#pragma omp parallel for
    for (int j = 0; j < j_size_; ++j) {
      new_matrix(j, i) = (*this)(i, j);
    }
  }
  double finish = clock();
  //std::cout << "Parallel runtime = " << (finish - start) / 1000.0 << std::endl;
  return new_matrix;
}



double ParallelMatrix::FrobeniusNorm() const {
  double start = clock();
  double norm = 0;
#pragma omp parallel for
  for (int i = 0; i < i_size_; ++i) {
#pragma omp parallel for reduction(+:norm)
    for (int j = 0; j < j_size_; ++j) {
      double a = (*this)(i, j);
      norm += a * a;
    }
  }
  norm = sqrt(norm);
  double finish = clock();
  //std::cout << "Parallel runtime = " << (finish - start) / 1000.0 << std::endl;
  return norm;
}



double ParallelMatrix::Determinant() const {
  const int n = i_size_;
  ParallelMatrix A(*this);

  for (int k = 0; k < n - 1; ++k) {
#pragma omp parallel for
    for (int i = k + 1; i < n; ++i) {
      double t = A(i, k) / A(k, k);
      for (int j = k + 1; j < n; ++j) {
        A(i, j) -= t * A(k, j);
      }
    }
  }
  double det = 1;
#pragma parallel for reduction(*:det)
  for (int i = 0; i < n; ++i) {
    det *= A(i, i);
  }
  return det;
}



ParallelMatrix ParallelMatrix::Invertible() const {
  const int n = i_size_;
  ParallelMatrix A(*this);
  ParallelMatrix I(n);
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    I(i, i) = 1;
  }
  for (int i = 0; i < n; ++i) {
    double t = A(i, i);
#pragma omp parallel for
    for (int j = 0; j < n; ++j) {
      A(i, j) /= t;
      I(i, j) /= t;
    }
#pragma omp parallel for
    for (int j = i + 1; j < n; ++j) {
      double t = A(j, i);
      for (int k = 0; k < n; ++k) {
        A(j, k) -= A(i, k) * t;
        I(j, k) -= I(i, k) * t;
      }
    }
  }
  for (int i = n - 1; i >= 0; --i) {
#pragma omp parallel for
    for (int j = i - 1; j >= 0; --j) {
      double t = A(j, i);
      for (int k = n - 1; k >= 0; --k) {
        A(j, k) -= A(i, k) * t;
        I(j, k) -= I(i, k) * t;
      }
    }
  }
  return I;
}

void ParallelMatrix::LUdecomposition(ParallelMatrix& L, ParallelMatrix& U) const {
  const int n = i_size_;
  U = *this;
  for (int k = 0; k < n - 1; ++k) {
#pragma omp parallel for
    for (int i = k + 1; i < n; ++i) {
      L(i, k) = U(i, k) / U(k, k);
      for (int j = k + 1; j < n; ++j) {
        U(i, j) -= L(i, k) * U(k, j);
      }
    }
  }
#pragma omp parallel for
  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      U(i, j) = 0;
    }
  }
}

double ParallelMatrix::SpectralNorm() {
  ParallelVector b;
  double norm = -1;
  for (int i = 0; i < RowNum(); ++i) {
    ParallelVector cur(ColumnNum());
    for (int j = 0; j < ColumnNum(); ++j) {
      cur[j] = (*this)(i, j);
    }
    double cur_norm = cur.Length();
    if (cur_norm > norm) {
      norm = cur_norm;
      b = cur;
    }
  }
  ParallelVector a;
  double eps = 1;
  while (eps > 1e-5) {
    ParallelVector newa = ((*this) * b) / (b * b);
    ParallelVector newb = (newa * (*this)) / (newa * newa);
    eps = std::fabs(a.Length() - newa.Length()) + fabs(b.Length() - newb.Length());
    a = newa;
    b = newb;
  }
  double res = a.Length() * b.Length();
  return res;
}



ptrdiff_t ParallelMatrix::RowNum() const noexcept {
  return i_size_;
}



ptrdiff_t ParallelMatrix::ColumnNum() const noexcept {
  return j_size_;
}



void ParallelMatrix::Resize(const ptrdiff_t i_new, const ptrdiff_t j_new) {
  if (i_new < 0 || j_new < 0) {
    throw std::length_error("Size cant't be < 0");
  }
  if (i_capacity_ < i_new || j_capacity_ < j_new) {
    Reserve(i_new, j_new);
  }
#pragma omp parallel for
  for (ptrdiff_t i = 0; i < i_size_; ++i) {
#pragma omp parallel for num_threads(j_new - j_size_)
    for (ptrdiff_t j = j_size_; j < j_new; ++j) {
      (*this)(i, j) = 0;
    }
  }
#pragma omp parallel for 
  for (ptrdiff_t i = i_size_; i < i_new; ++i) {
#pragma omp parallel for num_threads(j_new)
    for (ptrdiff_t j = 0; j < j_new; ++j) {
      (*this)(i, j) = 0;
    }
  }
  i_size_ = i_new;
  j_size_ = j_new;
}



void ParallelMatrix::Reserve(const ptrdiff_t i_new, const ptrdiff_t j_new) {
  if (i_new < 0 || j_new < 0) {
    throw std::length_error("Size cant't be < 0");
  }
  if (i_capacity_ < i_new || j_capacity_ < j_new) {
    ptrdiff_t i_new_cap(i_new);
    ptrdiff_t j_new_cap(j_new);
    if (i_new_cap < i_capacity_) {
      i_new_cap = i_capacity_;
    }
    if (j_new_cap < j_capacity_) {
      j_new_cap = j_capacity_;
    }
    double* new_data_ = new double[i_new_cap * j_new_cap];
    Copy(*this, new_data_, j_new_cap);
    delete[] data_;
    data_ = new_data_;
    i_capacity_ = i_new_cap;
    j_capacity_ = j_new_cap;
  }
}



void ParallelMatrix::Copy(const ParallelMatrix& from, double* const data,
  const ptrdiff_t j_cap) {
#pragma omp parallel for
  for (ptrdiff_t i = 0; i < from.i_size_; ++i) {
#pragma omp parallel for
    for (ptrdiff_t j = 0; j < from.j_size_; ++j) {
      *(data + i * j_cap + j) = from(i, j);
    }
  }
}



std::ostream& ParallelMatrix::WriteTo(std::ostream& ostrm) const {
  ostrm << '{';
  for (ptrdiff_t i = 0; i < i_size_; ++i) {
    ostrm << '{';
    for (ptrdiff_t j = 0; j < j_size_; ++j) {
      ostrm << (*this)(i, j);
      if (j < j_size_ - 1) {
        ostrm << ", ";
      }
    }
    ostrm << '}';
    if (i < i_size_ - 1) {
      ostrm << ", ";
    }
  }
  ostrm << '}';
  return ostrm;
}



ParallelMatrix operator+(const ParallelMatrix& lhs, const ParallelMatrix& rhs) {
  double start = clock();
  ParallelMatrix res(lhs);
  res += rhs;
  double finish = clock();
  //std::cout << "Parallel runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



ParallelMatrix operator-(const ParallelMatrix& lhs, const ParallelMatrix& rhs) {
  double start = clock();
  ParallelMatrix res(lhs);
  res -= rhs;
  double finish = clock();
  //std::cout << "Parallel runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



ParallelMatrix operator*(const double a, const ParallelMatrix & obj) {
  double start = clock();
  ParallelMatrix res(obj);
  res *= a;
  double finish = clock();
  //std::cout << "Parallel runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



ParallelMatrix operator*(const ParallelMatrix & obj, const double a) {
  return a * obj;
}



ParallelMatrix operator*(const ParallelMatrix& lhs, const ParallelMatrix& rhs) {
  double start = clock();
  if (lhs.ColumnNum() != rhs.RowNum()) {
    throw std::length_error("Number of columns in the left matrix and number of rows in the right"
      " one must be equal");
  }
  ParallelMatrix res(lhs.RowNum(), rhs.ColumnNum());
#pragma omp parallel for
  for (int i = 0; i < res.RowNum(); ++i) {
#pragma omp parallel for
    for (int j = 0; j < res.ColumnNum(); ++j) {
#pragma imp parallel for reduction(+:res(i, j))
      for (int k = 0; k < lhs.ColumnNum(); ++k) {
        res(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }
  double finish = clock();
  //std::cout << "Parallel runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



ParallelVector operator*(const ParallelMatrix& matr, const ParallelVector& vec) {
  double start = clock();
  if (matr.ColumnNum() != vec.Size()) {
    throw std::length_error("Number of columns in the matrix and number of rows in the vector"
      " must be equal");
  }
  ParallelVector res(matr.RowNum());
#pragma omp parallel for
  for (int i = 0; i < res.Size(); ++i) {
    double resi = 0;
#pragma omp parallel for reduction(+:resi)
    for (int k = 0; k < matr.RowNum(); ++k) {
      resi += matr(i, k) * vec[k];
    }
    res[i] = resi;
  }
  double finish = clock();
  //std::cout << "Parallel runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}

ParallelVector operator*(const ParallelVector & vec, const ParallelMatrix & matr) {
  double start = clock();
  if (matr.RowNum() != vec.Size()) {
    throw std::length_error("Number of columns in the matrix and number of rows in the vector"
      " must be equal");
  }
  ParallelVector res(matr.ColumnNum());
#pragma omp parallel for
  for (int i = 0; i < res.Size(); ++i) {
    for (int k = 0; k < matr.RowNum(); ++k) {
      res[i] += matr(k, i) * vec[k];
    }
  }
  double finish = clock();
  //std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}
