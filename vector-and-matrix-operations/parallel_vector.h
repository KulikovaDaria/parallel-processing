#ifndef PARALLEL_VECTOR_H
#define PARALLEL_VECTOR_H
#include <iosfwd>

class ParallelVector {
public:
  ParallelVector() = default;
  explicit ParallelVector(const ptrdiff_t size);
  ParallelVector(const ParallelVector& obj);
  ParallelVector(const std::initializer_list<double>& data);
  ~ParallelVector() noexcept;
  ParallelVector& operator=(const ParallelVector& obj);
  double& operator[](const ptrdiff_t i);
  double operator[](const ptrdiff_t i) const;
  ParallelVector& operator+=(const ParallelVector& obj);
  ParallelVector& operator-=(const ParallelVector& obj);
  ParallelVector& operator*=(const double a);
  double Length() const noexcept;
  ptrdiff_t Size() const noexcept;
  std::ostream& WriteTo(std::ostream& ostrm) const;

private:
  void Reserve(const ptrdiff_t new_capacity);
  void Resize(const ptrdiff_t new_size);
  static void Copy(const double* const first, const ptrdiff_t size,
    double* const data);

  ptrdiff_t size_{0};
  ptrdiff_t capacity_{0};
  double* data_{nullptr};
};


ParallelVector operator+(const ParallelVector& lhs, const ParallelVector& rhs);
ParallelVector operator-(const ParallelVector& lhs, const ParallelVector& rhs);
ParallelVector operator*(const double a, const ParallelVector& obj);
ParallelVector operator*(const ParallelVector& obj, const double a);
ParallelVector operator/(const ParallelVector& vec, const double d);
double operator*(const ParallelVector& lhs, const ParallelVector& rhs);
inline std::ostream& operator<<(std::ostream& ostrm, const ParallelVector& obj) {
  return obj.WriteTo(ostrm);
}

#endif