#ifndef MATRIX_H
#define MATRIX_H
#include "vector.h"
#include <iosfwd>

class Matrix {
public:
  Matrix() = default;
  Matrix(const ptrdiff_t size);
  Matrix(const ptrdiff_t i_size, const ptrdiff_t j_size);
  Matrix(const Matrix& obj);
  Matrix(const std::initializer_list<std::initializer_list<double>>& data);
  ~Matrix() noexcept;
  Matrix& operator=(const Matrix& obj);
  double& operator()(const ptrdiff_t i, const ptrdiff_t j);
  double operator()(const ptrdiff_t i, const ptrdiff_t j) const;
  Matrix& operator+=(const Matrix& obj);
  Matrix& operator-=(const Matrix& obj);
  Matrix& operator*=(const double a);
  Matrix& operator*=(const Matrix& obj);
  Matrix Transpose() const;
  double FrobeniusNorm() const;
  double Determinant() const;
  Matrix Invertible() const;
  void LUdecomposition(Matrix& L, Matrix& U) const;
  double SpectralNorm();
  ptrdiff_t RowNum() const noexcept;
  ptrdiff_t  ColumnNum() const noexcept;
  std::ostream& WriteTo(std::ostream& ostrm) const;

private:
  void Resize(const ptrdiff_t i_new, const ptrdiff_t j_new);
  void Reserve(const ptrdiff_t i_new, const ptrdiff_t j_new);
  static void Copy(const Matrix& from, double* const data,
    const ptrdiff_t j_cap);
  ptrdiff_t i_size_{0};
  ptrdiff_t j_size_{0};
  ptrdiff_t i_capacity_{0};
  ptrdiff_t j_capacity_{0};
  double* data_{nullptr};
};

Matrix operator+(const Matrix& lhs, const Matrix& rhs);
Matrix operator-(const Matrix& lhs, const Matrix& rhs);
Matrix operator*(const double a, const Matrix& obj);
Matrix operator*(const Matrix& obj, const double a);
Matrix operator*(const Matrix& lhs, const Matrix& rhs);
Vector operator*(const Matrix& matr, const Vector& vec);
Vector operator*(const Vector& vec, const Matrix& matr);
inline std::ostream& operator<<(std::ostream& ostrm, const Matrix& obj) {
  return obj.WriteTo(ostrm);
}

Vector GaussianElimination(Matrix A, Vector b);

#endif