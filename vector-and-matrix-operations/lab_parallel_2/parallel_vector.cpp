#include <ctime>
#include <exception>
#include <math.h>
#include <omp.h>
#include <sstream>
#include "parallel_vector.h"

ParallelVector::ParallelVector(const ptrdiff_t size)
  : size_(size), capacity_(size) {
  if (size < 0) {
    throw std::length_error("Size can't be < 0");
  }
  data_ = new double[size];
#pragma omp parallel
  for (ptrdiff_t i = 0; i < size; ++i) {
    (*this)[i] = 0;
  }
}



ParallelVector::ParallelVector(const ParallelVector& obj)
  :size_(obj.size_), capacity_(obj.size_), data_(new double[obj.size_]) {
  Copy(obj.data_, obj.size_, data_);
}



ParallelVector::ParallelVector(const std::initializer_list<double>& data)
  :ParallelVector(data.size()) {
  int i = 0;
  for (double element : data) {
    (*this)[i++] = element;
  }
}



ParallelVector::~ParallelVector() noexcept {
  delete[] data_;
  data_ = nullptr;
}



void ParallelVector::Reserve(const ptrdiff_t new_capacity) {
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



void ParallelVector::Resize(const ptrdiff_t new_size) {
  if (new_size < 0) {
    throw std::length_error("Size can't be < 0");
  }
  if (capacity_ < new_size) {
    Reserve(new_size);
  }
  size_ = new_size;
}



ParallelVector& ParallelVector::operator=(const ParallelVector& obj) {
  if (this != &obj) {
    if (capacity_ < obj.size_) {
      Resize(obj.size_);
    }
    Copy(obj.data_, obj.size_, data_);
    size_ = obj.size_;
  }
  return *this;
}



double& ParallelVector::operator[](const ptrdiff_t i) {
  if ((i < 0) || (i >= size_)) {
    throw std::out_of_range("Invalid index");
  }
  return *(data_ + i);
}



double ParallelVector::operator[](const ptrdiff_t i) const {
  if ((i < 0) || (i >= size_)) {
    throw std::out_of_range("Invalid index");
  }
  return *(data_ + i);
}



ParallelVector& ParallelVector::operator+=(const ParallelVector& obj) {
  if (size_ != obj.size_) {
    throw std::length_error("Vectors must be the equal size");
  }
#pragma omp parallel for
  for (int i = 0; i < size_; ++i) {
    (*this)[i] += obj[i];
  }
  return *this;
}



ParallelVector& ParallelVector::operator-=(const ParallelVector& obj) {
  if (size_ != obj.size_) {
    throw std::length_error("Vectors must be the equal size");
  }
#pragma omp parallel for
  for (int i = 0; i < size_; ++i) {
    (*this)[i] -= obj[i];
  }
  return *this;
}



ParallelVector& ParallelVector::operator*=(const double a) {
#pragma omp parallel for num_threads(size_)
  for (int i = 0; i < size_; ++i) {
    (*this)[i] *= a;
  }
  return *this;
}



double ParallelVector::Length() const noexcept {
  double start = clock();
  double res = 0;
#pragma omp parallel for reduction(+:res)
  for (int i = 0; i < size_; ++i) {
    double a = (*this)[i];
    res += a * a;
  }
  double finish = clock();
  res = sqrt(res);
  //std::cout << "Parallel runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



ptrdiff_t ParallelVector::Size() const noexcept {
  return size_;
}



void ParallelVector::Copy(const double* const first, const ptrdiff_t size,
  double* const data) {
#pragma omp parallel for
  for (ptrdiff_t i = 0; i < size; ++i) {
    *(data + i) = *(first + i);
  }
}



std::ostream& ParallelVector::WriteTo(std::ostream& ostrm) const {
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



ParallelVector operator+(const ParallelVector & lhs, const ParallelVector & rhs) {
  double start = clock();
  ParallelVector res(lhs);
  res += rhs;
  double finish = clock();
  //std::cout << "Parallel runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



ParallelVector operator-(const ParallelVector & lhs, const ParallelVector & rhs) {
  double start = clock();
  ParallelVector res(lhs);
  res -= rhs;
  double finish = clock();
  //std::cout << "Parallel runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



ParallelVector operator*(const double a, const ParallelVector& obj) {
  double start = clock();
  ParallelVector res(obj);
  res *= a;
  double finish = clock();
  //std::cout << "Parallel runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}



ParallelVector operator*(const ParallelVector& obj, const double a) {
  return a * obj;
}

ParallelVector operator/(const ParallelVector & vec, const double d) {
  ParallelVector res(vec);
#pragma omp parallel for
  for (int i = 0; i < res.Size(); ++i) {
    res[i] /= d;
  }
  return res;
}



double operator*(const ParallelVector& lhs, const ParallelVector& rhs) {
  double start = clock();
  double res = 0;
#pragma omp parallel for reduction(+:res)
  for (int i = 0; i < lhs.Size(); ++i) {
    res += lhs[i] * rhs[i];
  }
  double finish = clock();
  //std::cout << "Parallel runtime = " << (finish - start) / 1000.0 << std::endl;
  return res;
}
