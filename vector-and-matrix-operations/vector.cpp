#include <ctime>
#include <exception>
#include <math.h>
#include <sstream>
#include "vector.h"

Vector::Vector(const ptrdiff_t size)
  : size_(size), capacity_(size) {
  if (size < 0) {
    throw std::length_error("Size can't be < 0");
  }
  data_ = new double[size];
  for (ptrdiff_t i = 0; i < size; ++i) {
    (*this)[i] = 0;
  }
}



Vector::Vector(const Vector& obj)
  :size_(obj.size_), capacity_(obj.size_), data_(new double[obj.size_]) {
  Copy(obj.data_, obj.size_, data_);
}



Vector::Vector(const std::initializer_list<double>& data)
  :Vector(data.size()) {
  int i = 0;
  for (double element : data) {
    (*this)[i++] = element;
  }
}



Vector::~Vector() noexcept {
  delete[] data_;
  data_ = nullptr;
}



void Vector::Reserve(const ptrdiff_t new_capacity) {
  if (new_capacity < 0) {
    throw std::length_error("Size can't be < 0");
  }
  if (capacity_ < new_capacity) {
    double* new_data(new double[new_capacity]);
    Copy(data_, size_, new_data);
    delete[] data_;
    data_ = new_data;
    capacity_ = new_capacity;
  }
}



void Vector::Resize(const ptrdiff_t new_size) {
  if (new_size < 0) {
    throw std::length_error("Size can't be < 0");
  }
  if (capacity_ < new_size) {
    Reserve(new_size);
  }
  size_ = new_size;
}



Vector& Vector::operator=(const Vector& obj) {
  if (this != &obj) {
    if (capacity_ < obj.size_) {
      Resize(obj.size_);
    }
    Copy(obj.data_, obj.size_, data_);
    size_ = obj.size_;
  }
  return *this;
}



double& Vector::operator[](const ptrdiff_t i) {
  if ((i < 0) || (i >= size_)) {
    throw std::out_of_range("Invalid index");
  }
  return *(data_ + i);
}



double Vector::operator[](const ptrdiff_t i) const {
  if ((i < 0) || (i >= size_)) {
    throw std::out_of_range("Invalid index");
  }
  return *(data_ + i);
}



Vector& Vector::operator+=(const Vector& obj) {
  if (size_ != obj.size_) {
    throw std::length_error("Vectors must be the equal size");
  }
  for (int i = 0; i < size_; ++i) {
    (*this)[i] += obj[i];
  }
  return *this;
}



Vector& Vector::operator-=(const Vector& obj) {
  if (size_ != obj.size_) {
    throw std::length_error("Vectors must be the equal size");
  }
  for (int i = 0; i < size_; ++i) {
    (*this)[i] -= obj[i];
  }
  return *this;
}



Vector& Vector::operator*=(const double a) {
  for (int i = 0; i < size_; ++i) {
    (*this)[i] *= a;
  }
  return *this;
}



double Vector::Length() const noexcept {
  double start = clock();
  double res = 0;
  for (int i = 0; i < size_; ++i) {
    double a = (*this)[i];
    res += a * a;
  }
  double finish = clock();
  res = sqrt(res);
  //std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



ptrdiff_t Vector::Size() const noexcept {
  return size_;
}



void Vector::Copy(const double* const first, const ptrdiff_t size,
  double* const data) {
  for (ptrdiff_t i = 0; i < size; ++i) {
    *(data + i) = *(first + i);
  }
}



std::ostream& Vector::WriteTo(std::ostream& ostrm) const {
  ostrm << '{';
  for (ptrdiff_t i = 0; i < size_; ++i) {
    ostrm << (*this)[i];
    if (i < size_ - 1) {
      ostrm << ", ";
    }
  }
  ostrm << '}';
  return ostrm;
}



Vector operator+(const Vector & lhs, const Vector & rhs) {
  double start = clock();
  Vector res(lhs);
  res += rhs;
  double finish = clock();
  //std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



Vector operator-(const Vector & lhs, const Vector & rhs) {
  double start = clock();
  Vector res(lhs);
  res -= rhs;
  double finish = clock();
  //std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



Vector operator*(const double a, const Vector& obj)  {
  double start = clock();
  Vector res(obj);
  res *= a;
  double finish = clock();
  //std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



Vector operator*(const Vector& obj, const double a) {
  return a * obj;
}



double operator*(const Vector& lhs, const Vector& rhs) {
  double start = clock();
  double res = 0;
  for (int i = 0; i < lhs.Size(); ++i) {
    res += lhs[i] * rhs[i];
  }
  double finish = clock();
 // std::cout << "Sequential runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}

Vector operator/(const Vector & vec, const double d) {
  Vector res(vec);
  for (int i = 0; i < res.Size(); ++i) {
    res[i] /= d;
  }
  return res;
}
