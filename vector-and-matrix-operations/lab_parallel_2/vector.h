#ifndef VECTOR_H
#define VECTOR_H
#include <iosfwd>

class Vector {
public:
  Vector() = default;
  explicit Vector(const ptrdiff_t size);
  Vector(const Vector& obj);
  Vector(const std::initializer_list<double>& data);
  ~Vector() noexcept;
  Vector& operator=(const Vector& obj);
  double& operator[](const ptrdiff_t i);  
  double operator[](const ptrdiff_t i) const;
  Vector& operator+=(const Vector& obj);
  Vector& operator-=(const Vector& obj);
  Vector& operator*=(const double a);
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


Vector operator+(const Vector& lhs, const Vector& rhs);
Vector operator-(const Vector& lhs, const Vector& rhs);
Vector operator*(const double a, const Vector& obj);
Vector operator*(const Vector& obj, const double a);
double operator*(const Vector& lhs, const Vector& rhs);
Vector operator/(const Vector& vec, const double d);
inline std::ostream& operator<<(std::ostream& ostrm, const Vector& obj) {
  return obj.WriteTo(ostrm);
}

#endif