#include "matrix.h"
#include <ctime>
#include <exception>
#include <math.h>
#include <sstream>

Matrix::Matrix(const ptrdiff_t size)
  :Matrix(size, size) {}



Matrix::Matrix(const ptrdiff_t i_size, const ptrdiff_t j_size)
  : i_size_(i_size), j_size_(j_size), i_capacity_(i_size),
  j_capacity_(j_size) {
  if (i_size < 0 || j_size < 0) {
    throw std::length_error("Size cant't be < 0");
  }
  data_ = new double[i_size * j_size];
  for (ptrdiff_t i = 0; i < i_size * j_size; ++i) {
    *(data_ + i) = 0;
  }
}



Matrix::Matrix(const Matrix& obj)
  : i_size_(obj.i_size_), j_size_(obj.j_size_), i_capacity_(obj.i_size_),
  j_capacity_(obj.j_size_), data_(new double[obj.i_size_ * obj.j_size_]) {
  Copy(obj, data_, j_capacity_);
}



Matrix::Matrix(const std::initializer_list<std::initializer_list<double>>& data)
  :Matrix(data.size(), data.begin()->size()) {
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



Matrix::~Matrix() noexcept {
  delete[] data_;
  data_ = nullptr;
}



Matrix& Matrix::operator=(const Matrix& obj) {
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



double& Matrix::operator()(const ptrdiff_t i, const ptrdiff_t j) {
  return *(data_ + i * j_capacity_ + j);
}



double Matrix::operator()(const ptrdiff_t i, const ptrdiff_t j) const {
  return *(data_ + i * j_capacity_ + j);
}



Matrix& Matrix::operator+=(const Matrix& obj) {
  if (i_size_ != obj.i_size_ || j_size_ != obj.j_size_) {
    throw std::length_error("Matrixes must be the equal size");
  }
  for (int i = 0; i < i_size_; ++i) {
    for (int j = 0; j < j_size_; ++j) {
      (*this)(i, j) += obj(i, j);
    }
  }
  return *this;
}



Matrix& Matrix::operator-=(const Matrix& obj) {
  if (i_size_ != obj.i_size_ || j_size_ != obj.j_size_) {
    throw std::length_error("Matrixes must be the equal size");
  }
  for (int i = 0; i < i_size_; ++i) {
    for (int j = 0; j < j_size_; ++j) {
      (*this)(i, j) -= obj(i, j);
    }
  }
  return *this;
}



Matrix& Matrix::operator*=(const double a) {
  for (int i = 0; i < i_size_; ++i) {
    for (int j = 0; j < j_size_; ++j) {
      (*this)(i, j) *= a;
    }
  }
  return *this;
}



Matrix& Matrix::operator*=(const Matrix& obj) {
  *this = *this * obj;
  return *this;
}



Matrix Matrix::Transpose() const {
  double start = clock();
  Matrix new_matrix(j_size_, i_size_);
  for (int i = 0; i < i_size_; ++i) {
    for (int j = 0; j < j_size_; ++j) {
      new_matrix(j, i) = (*this)(i, j);
    }
  }
  double finish = clock();
 // std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return new_matrix;
}



double Matrix::FrobeniusNorm() const {
  double start = clock();
  double norm = 0;
  for (int i = 0; i < i_size_; ++i) {
    for (int j = 0; j < j_size_; ++j) {
      double a = (*this)(i, j);
      norm += a * a;
    }
  }
  norm = sqrt(norm);
  double finish = clock();
  //std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return norm;
}



double Matrix::Determinant() const {
  const int n = i_size_;
  Matrix A(*this);

  for (int k = 0; k < n - 1; ++k) {
    for (int i = k + 1; i < n; ++i) {
      double t = A(i,k) / A(k, k);
      for (int j = k + 1; j < n; ++j) {
        A(i, j) -= t * A(k, j);
      }
    }
  }
  double det = 1;
  for (int i = 0; i < n; ++i) {
    det *= A(i, i);
  }
  return det;
}



Matrix Matrix::Invertible() const {
  const int n = i_size_;
  Matrix A(*this);
  Matrix I(n);
  for (int i = 0; i < n; ++i) {
    I(i, i) = 1;
  }
  for (int i = 0; i < n; ++i) {
    double t = A(i, i);
    for (int j = 0; j < n; ++j) {
      A(i, j) /= t;
      I(i, j) /= t;
    }
    for (int j = i + 1; j < n; ++j) {
      double t = A(j, i);
      for (int k = 0; k < n; ++k) {
        A(j, k) -= A(i, k) * t;
        I(j, k) -= I(i, k) * t;
      }
    }
  }
  for (int i = n - 1; i >= 0; --i) {
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

void Matrix::LUdecomposition(Matrix& L, Matrix& U) const {
  const int n = i_size_;
  U = *this;

  for (int k = 0; k < n - 1; ++k) {
    for (int i = k + 1; i < n; ++i) {
      L(i, k) = U(i, k) / U(k, k);
      for (int j = k + 1; j < n; ++j) {
        U(i, j) -= L(i, k) * U(k, j);
      }
    }
  }
  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      U(i, j) = 0;
    }
  }
}

double Matrix::SpectralNorm() {
  Vector b;
  double norm = -1;
  for (int i = 0; i < RowNum(); ++i) {
    Vector cur(ColumnNum());
    for (int j = 0; j < ColumnNum(); ++j) {
      cur[j] = (*this)(i, j);
    }
    double cur_norm = cur.Length();
    if (cur_norm > norm) {
      norm = cur_norm;
      b = cur;
    }
  }
  Vector a;
  double eps = 1;
  while (eps > 1e-5) {
    Vector newa = ((*this) * b) / (b * b);
    Vector newb = (newa * (*this)) / (newa * newa);
    eps = std::fabs(a.Length() - newa.Length()) + fabs(b.Length() - newb.Length());
    a = newa;
    b = newb;
  }
  double res = a.Length() * b.Length();
  return res;
}



ptrdiff_t Matrix::RowNum() const noexcept {
  return i_size_;
}



ptrdiff_t Matrix::ColumnNum() const noexcept {
  return j_size_;
}



void Matrix::Resize(const ptrdiff_t i_new, const ptrdiff_t j_new) {
  if (i_new < 0 || j_new < 0) {
    throw std::length_error("Size cant't be < 0");
  }
  if (i_capacity_ < i_new || j_capacity_ < j_new) {
    Reserve(i_new, j_new);
  }
  for (ptrdiff_t i = 0; i < i_size_; ++i) {
    for (ptrdiff_t j = j_size_; j < j_new; ++j) {
      (*this)(i, j) = 0;
    }
  }
  for (ptrdiff_t i = i_size_; i < i_new; ++i) {
    for (ptrdiff_t j = 0; j < j_new; ++j) {
      (*this)(i, j) = 0;
    }
  }
  i_size_ = i_new;
  j_size_ = j_new;
}



void Matrix::Reserve(const ptrdiff_t i_new, const ptrdiff_t j_new) {
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



void Matrix::Copy(const Matrix& from, double* const data,
  const ptrdiff_t j_cap) {
  for (ptrdiff_t i = 0; i < from.i_size_; ++i) {
    for (ptrdiff_t j = 0; j < from.j_size_; ++j) {
      *(data + i * j_cap + j) = from(i, j);
    }
  }
}



std::ostream& Matrix::WriteTo(std::ostream& ostrm) const {
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



Matrix operator+(const Matrix& lhs, const Matrix& rhs) {
  double start = clock();
  Matrix res(lhs);
  res += rhs;
  double finish = clock();
  //std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



Matrix operator-(const Matrix& lhs, const Matrix& rhs) {
  double start = clock();
  Matrix res(lhs);
  res -= rhs;
  double finish = clock();
  //std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



Matrix operator*(const double a, const Matrix & obj) {
  double start = clock();
  Matrix res(obj);
  res *= a;
  double finish = clock();
  //std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



Matrix operator*(const Matrix & obj, const double a) {
  return a * obj;
}



Matrix operator*(const Matrix& lhs, const Matrix& rhs) {
  double start = clock();
  if (lhs.ColumnNum() != rhs.RowNum()) {
    throw std::length_error("Number of columns in the left matrix and number of rows in the right"
      " one must be equal");
  }
  Matrix res(lhs.RowNum(), rhs.ColumnNum());
  for (int i = 0; i < res.RowNum(); ++i) {
    for (int j = 0; j < res.ColumnNum(); ++j) {
      for (int k = 0; k < lhs.ColumnNum(); ++k) {
        res(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }
  double finish = clock();
  //std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



Vector operator*(const Matrix& matr, const Vector& vec) {
  double start = clock();
  if (matr.ColumnNum() != vec.Size()) {
    throw std::length_error("Number of columns in the matrix and number of rows in the vector"
      " must be equal");
  }
  Vector res(matr.RowNum());
  for (int i = 0; i < res.Size(); ++i) {
    for (int k = 0; k < matr.ColumnNum(); ++k) {
      res[i] += matr(i, k) * vec[k];
    }
  }
  double finish = clock();
  //std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



Vector operator*(const Vector& vec, const Matrix& matr) {
  double start = clock();
  if (matr.RowNum() != vec.Size()) {
    throw std::length_error("Number of columns in the matrix and number of rows in the vector"
      " must be equal");
  }
  Vector res(matr.ColumnNum());
  for (int i = 0; i < res.Size(); ++i) {
    for (int k = 0; k < matr.RowNum(); ++k) {
      res[i] += matr(k, i) * vec[k];
    }
  }
  double finish = clock();
  //std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}