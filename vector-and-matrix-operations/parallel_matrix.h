#ifndef PARALLEL_MATRIX_H
#define PARALLEL_MATRIX_H
#include "parallel_vector.h"
#include <iosfwd>

class ParallelMatrix {
public:
  ParallelMatrix() = default;
  ParallelMatrix(const ptrdiff_t size);
  ParallelMatrix(const ptrdiff_t i_size, const ptrdiff_t j_size);
  ParallelMatrix(const ParallelMatrix& obj);
  ParallelMatrix(const std::initializer_list<std::initializer_list<double>>& data);
  ~ParallelMatrix() noexcept;
  ParallelMatrix& operator=(const ParallelMatrix& obj);
  double& operator()(const ptrdiff_t i, const ptrdiff_t j);
  double operator()(const ptrdiff_t i, const ptrdiff_t j) const;
  ParallelMatrix& operator+=(const ParallelMatrix& obj);
  ParallelMatrix& operator-=(const ParallelMatrix& obj);
  ParallelMatrix& operator*=(const double a);
  ParallelMatrix& operator*=(const ParallelMatrix& obj);
  ParallelMatrix Transpose() const;
  double Determinant() const;
  ParallelMatrix Invertible() const;
  void LUdecomposition(ParallelMatrix& L, ParallelMatrix& U) const;
  double SpectralNorm();
  double FrobeniusNorm() const;
  ptrdiff_t RowNum() const noexcept;
  ptrdiff_t  ColumnNum() const noexcept;
  std::ostream& WriteTo(std::ostream& ostrm) const;

private:
  void Resize(const ptrdiff_t i_new, const ptrdiff_t j_new);
  void Reserve(const ptrdiff_t i_new, const ptrdiff_t j_new);
  static void Copy(const ParallelMatrix& from, double* const data,
    const ptrdiff_t j_cap);
  ptrdiff_t i_size_{0};
  ptrdiff_t j_size_{0};
  ptrdiff_t i_capacity_{0};
  ptrdiff_t j_capacity_{0};
  double* data_{nullptr};
};

ParallelMatrix operator+(const ParallelMatrix& lhs, const ParallelMatrix& rhs);
ParallelMatrix operator-(const ParallelMatrix& lhs, const ParallelMatrix& rhs);
ParallelMatrix operator*(const double a, const ParallelMatrix& obj);
ParallelMatrix operator*(const ParallelMatrix& obj, const double a);
ParallelMatrix operator*(const ParallelMatrix& lhs, const ParallelMatrix& rhs);
ParallelVector operator*(const ParallelMatrix& matr, const ParallelVector& vec);
ParallelVector operator*(const ParallelVector& vec, const ParallelMatrix& matr);
inline std::ostream& operator<<(std::ostream& ostrm, const ParallelMatrix& obj) {
  return obj.WriteTo(ostrm);
}

ParallelVector GaussianElimination(ParallelMatrix A, ParallelVector b);

#endif