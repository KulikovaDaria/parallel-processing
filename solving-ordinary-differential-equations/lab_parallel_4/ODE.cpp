#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>

std::vector<double> operator+(const std::vector<double>& lhs,
  const std::vector<double>& rhs) {
  std::vector<double> res(lhs.size(), 0);
  for (int i = 0; i < res.size(); ++i) {
    res[i] = lhs[i] + rhs[i];
  }
  return res;
}



// Функтор, возвращающий значение правой части ОДУ
// при заданных значениях t и вектора y
class FunctorODE {
public:
  FunctorODE(const std::vector<double>& k)
    :k(k) {}
  double operator()(const double t, const std::vector<double>& y) const {
    double res = k[y.size()] * cos(t);
    for (int i = 0; i < y.size(); ++i) {
      res += k[i] * y[i];
    }
    return res;
  }
private:
  std::vector<double> k{0, 0};
};



// Функтор, возвращающий значение решения в точке t
class FunctorAns {
public:
  FunctorAns(const std::vector<double>& p, const std::vector<double>& k,
    const std::vector<double>& a)
    :p(p), k(k), a(a) {}
  double operator()(const double t) const {
    double res = k[0] * exp(p[0] * t) * sin(a[0] * t) + k[1] * exp(p[1] * t)
      * cos(a[1] * t) + k[2] * exp(p[2] * t) + k[3] * exp(p[3] * t);
    return res;
  }
private:
  std::vector<double> p{0, 0, 0, 0};
  std::vector<double> k{0, 0, 0, 0};
  std::vector<double> a{0, 0};
};



// Численное интегрирования методом трапеций
std::vector<double> Integral(const double t1, const double t2,
  const std::vector<double>& y1, const std::vector<double>& y2,
  const std::vector<FunctorODE>& f) {

  const int n = f.size();
  std::vector<double> I(n, 0);
  const double k = (t2 - t1) / 2;
  for (int i = 0; i < n; ++i) {
    I[i] = k * (f[i](t1, y1) + f[i](t2, y2));
  }
  return I;
}



// Решение задачи Коши для системы ОДУ методом Пикара
std::vector<std::vector<double>> PicardMethod(const std::vector<double>& y0,
  const double t0, const double tm, const std::vector<FunctorODE>& f, const int m) {

  // n - зазмерность системы
  const int n = y0.size();
  // h - расстояние между двумя соседними точками
  const double h = (tm - t0) / m;
  // Значение функции y в (m + 1) точке отрезка [t0, tm]
  std::vector<std::vector<double>> y(m + 1, std::vector<double>(n, 0));
  // Значения y, полученные на предыдущей итерации
  std::vector<std::vector<double>> prev_y(m + 1, std::vector<double>(n, 0));
  // Инициализируем вектор y в первом приближении значением y0 в каждой точке
  for (int i = 0; i <= m; ++i) {
    y[i] = y0;
  }
  // d - изменение значения y[m]
  double d = 0;
  do {
    swap(prev_y, y);
    double t = t0;
    // Рассчитаем значения функции y в каждой точке, используя численной
    // интегрирование и значения y, полученные на предыдущей итерации
    y[0] = y0;
    for (int i = 1; i <= m; ++i, t += h) {
      y[i] = y[i - 1] + Integral(t, t + h, prev_y[i - 1], prev_y[i], f);
    }
    d = 0;
    // Рассчитаем изменение значения y[m]
    // Итерационнй процесс прекращается при малом значение d (<= 1e-7)
    for (int i = 0; i < n; ++i) {
      d += fabs(prev_y[m][i] - y[m][i]);
    }
  } while (d > 1e-7);
  return y;
}



// Решение задачи Коши для системы ОДУ расспараллеленным методом Пикара
std::vector<std::vector<double>> ParallelPicardMethod(const std::vector<double>& y0,
  const double t0, const double tm, std::vector<FunctorODE>& f, const int m) {

  // n - размерность системы
  const int n = y0.size();
  // Для параллельного вычисления разобьем отрезок [t0, tm] на p блоков
  const int p = 8;
  // h - расстояние между двумя соседними точками
  const double h = (tm - t0) / m;
  // Значение функции y в (m + 1) точке отрезка [t0, tm]
  std::vector<std::vector<double>> y(m + 1, std::vector<double>(n, 0));
  // Значения y, полученные на предыдущей итерации
  std::vector<std::vector<double>> prev_y(m + 1, std::vector<double>(n, 0));
  // Инициализируем вектор y в первом приближении значением y0 в каждой точке
#pragma omp parallel for
  for (int i = 0; i <= m; ++i) {
    y[i] = y0;
  }
  // d - изменение значения y[m]
  double d = 0;
  do {
    swap(prev_y, y);
    double t = t0;
    // Вектор значений интегралов в каждой из точек
    std::vector<std::vector<double>> I(m, std::vector<double>(n, 0));
    // Сумма интегралов внутри каждого блока
    std::vector<std::vector<double>> sumI(p, std::vector<double>(n, 0));
    // Рассчитаем значения интегралов в каждой точке и суммы интегралов
    // внутри блока, параллельно для каждого блока
#pragma omp parallel for
    for (int mu = 0; mu < p; ++mu) {
      for (int l = 0; l < m / p; ++l) {
        int idx = l + mu * m / p;
        I[idx] = Integral(t0 + h * idx, t0 + h * (idx + 1),
          prev_y[idx], prev_y[idx + 1], f);
        sumI[mu] = sumI[mu] + I[idx];
      }
    }
    // Перекинем полученные суммы интегралов во все последующие блоки
    for (int mu = 1; mu < p; ++mu) {
      sumI[mu] = sumI[mu] + sumI[mu - 1];
    }
    y[0] = y0;
    // Параллельно рассчитаем первые значения функции y в каждом из блоков,
    // используя полученные суммы интегралов
#pragma omp parallel for
    for (int mu = 1; mu <= p; ++mu) {
      y[mu * m / p] = y0 + sumI[mu - 1];
    }
    // Рассчитаем остальные значения y параллельно для каждого блока
#pragma omp parallel for
    for (int mu = 0; mu < p; ++mu) {
      for (int l = 0; l < m / p - 1; ++l) {
        int idx = l + mu * m / p;
        y[idx + 1] = y[idx] + I[idx];
      }
    }
    d = 0;
    // Рассчитаем изменение значения y[m]
    // Итерационнй процесс прекращается при малом значение d (<= 1e-7)
#pragma omp parallel for reduction(+:d)
    for (int i = 0; i < n; ++i) {
      d += fabs(prev_y[m][i] - y[m][i]);
    }
  } while (d > 1e-7);
  return y;
}



std::vector<std::vector<double>> RungeKuttaMethod(const std::vector<double>& y0,
  const double t0, const double tm, const std::vector<FunctorODE>& f, const int m) {

  const int n = y0.size();
  const double h = (tm - t0) / m;
  int numk = 4;
  std::vector<std::vector<double>> k(numk, std::vector<double>(n, 0));
  std::vector<std::vector<double>> y(m + 1, std::vector<double>(n, 0));
  y[0] = y0;
  double t = t0;
  for (int s = 0; s < m; t += h, s++) {
    std::vector<double> cur_y(y[s]);
    double cur_t = t;
    std::vector<double> dt{0, h / 2, h / 2, h};
    for (int j = 0; j < numk; ++j) {
      cur_t = t + dt[j];
      if (j > 0) {
        for (int i = 0; i < n; ++i) {
          cur_y[i] = y[s][i] + dt[j] * k[j - 1][i];
        }
      }
      for (int i = 0; i < n; ++i) {
        k[j][i] = f[i](cur_t, cur_y);
      }
    }
    for (int i = 0; i < n; ++i) {
      y[s + 1][i] = y[s][i] + h * (k[0][i] + 2 * k[1][i] + 2 * k[2][i] + k[3][i]) / 6.0;
    }
  }
  return y;
}

std::vector<std::vector<double>> ParallelRungeKuttaMethod(const std::vector<double>& y0,
  const double t0, const double tm, const std::vector<FunctorODE>& f, const int m) {

  const int n = y0.size();
  const double h = (tm - t0) / m;
  int numk = 4;
  std::vector<std::vector<double>> k(numk, std::vector<double>(n, 0));
  std::vector<std::vector<double>> y(m + 1, std::vector<double>(n, 0));
  y[0] = y0;
  double t = t0;
  for (int s = 0; s < m; t += h, s++) {
    std::vector<double> cur_y(y[s]);
    double cur_t = t;
    std::vector<double> dt{0, h / 2, h / 2, h};
    for (int j = 0; j < numk; ++j) {
      cur_t = t + dt[j];
      if (j > 0) {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
          cur_y[i] = y[s][i] + dt[j] * k[j - 1][i];
        }
      }
#pragma omp parallel for
      for (int i = 0; i < n; ++i) {
        k[j][i] = f[i](cur_t, cur_y);
      }
    }
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      y[s + 1][i] = y[s][i] + h * (k[0][i] + 2 * k[1][i] + 2 * k[2][i] + k[3][i]) / 6.0;
    }
  }
  return y;
}



std::vector<std::vector<double>> PredictorCorrectorMethod(const std::vector<double>& y0,
  const double t0, const double tm, const std::vector<FunctorODE>& f, const int m) {

  const int n = y0.size();
  const double h = (tm - t0) / m;

  double t = t0 + h * 3;
  int m1 = 3;
  std::vector<std::vector<double>> y = RungeKuttaMethod(y0, t0, t, f, m1);
  y.resize(m + 1);

  for (int i = 3; i < m; ++i, t += h) {
    y[i + 1].resize(n);
    std::vector<double> y1(n), f1(n);
    for (int j = 0; j < n; ++j) {
      y1[j] = y[i][j] + h * (55 * f[j](t, y[i]) - 59 * f[j](t - h, y[i - 1])
        + 37 * f[j](t - 2 * h, y[i - 2]) - 9 * f[j](t - 3 * h, y[i - 3])) / 24.0;
    }
    for (int j = 0; j < n; ++j) {
      f1[j] = f[j](t + h, y1);
    }
    for (int j = 0; j < n; ++j) {
      y[i + 1][j] = y[i][j] + h * (9 * f1[j] + 19 * f[j](t, y[i]) - 5 * f[j](t - h, y[i - 1]) + f[j](t - 2 * h, y[i - 2])) / 24.0;
    }
  }
  return y;
}

std::vector<std::vector<double>> ParallelPredictorCorrectorMethod(const std::vector<double>& y0,
  const double t0, const double tm, const std::vector<FunctorODE>& f, const int m) {

  const int n = y0.size();
  const double h = (tm - t0) / m;

  double t = t0 + h * 3;
  int m1 = 3;
  std::vector<std::vector<double>> y = ParallelRungeKuttaMethod(y0, t0, t, f, m1);
  y.resize(m + 1);

  for (int i = 3; i < m; ++i, t += h) {
    y[i + 1].resize(n);
    std::vector<double> y1(n), f1(n);
#pragma omp parallel for
    for (int j = 0; j < n; ++j) {
      y1[j] = y[i][j] + h * (55 * f[j](t, y[i]) - 59 * f[j](t - h, y[i - 1])
        + 37 * f[j](t - 2 * h, y[i - 2]) - 9 * f[j](t - 3 * h, y[i - 3])) / 24.0;
    }
#pragma omp parallel for
    for (int j = 0; j < n; ++j) {
      f1[j] = f[j](t + h, y1);
    }
#pragma omp parallel for
    for (int j = 0; j < n; ++j) {
      y[i + 1][j] = y[i][j] + h * (9 * f1[j] + 19 * f[j](t, y[i]) - 5 * f[j](t - h, y[i - 1]) + f[j](t - 2 * h, y[i - 2])) / 24.0;
    }
  }
  return y;
}

int main() {

  setlocale(LC_ALL, "Russian");
  std::cout << std::setprecision(6);

  // Размерность системы
  const int n = 5;
  // Инициализируем вектор функторов для правой части системы ОДУ
  std::vector<FunctorODE> f{{{1, -1, -1, 0, 0, 0}},
                            {{1, 1, 0, 0, 0, 0}},
                            {{3, 0, 1, 0, 0, 0}},
                            {{0, 0, 0, 0, 1, -5}},
                            {{0, 0, 0, 2, 1, 0}}};
  // Начальное условие
  std::vector<double> y0{2, 2, 2, 1, 4};
  // Рассматриваемый отрезок
  double t0 = 0, tm = 1;

  // Время последовательного и параллельного рассчета
  double seq_time = 0, rk_seq_time = 0, pc_seq_time = 0;
  double par_time = 0, rk_par_time = 0, pc_par_time = 0;
  std::vector<std::vector<double>> seq_ans, par_ans, rk_seq_ans, rk_par_ans, pc_seq_ans, pc_par_ans;
  const int m = 100000;
  // Для более точного определения коэффицианта ускорения запустим методы q раз
  int q = 1;
  while (q--) {
    double start = clock();
    seq_ans = PicardMethod(y0, t0, tm, f, m);
    double finish = clock();
    seq_time += finish - start;

    start = clock();
    par_ans = ParallelPicardMethod(y0, t0, tm, f, m);
    finish = clock();
    par_time += finish - start;

    start = clock();
    rk_seq_ans = RungeKuttaMethod(y0, t0, tm, f, m);
    finish = clock();
    rk_seq_time += finish - start;

    start = clock();
    rk_par_ans = ParallelRungeKuttaMethod(y0, t0, tm, f, m);
    finish = clock();
    rk_par_time += finish - start;

    start = clock();
    pc_seq_ans = PredictorCorrectorMethod(y0, t0, tm, f, m);
    finish = clock();
    pc_seq_time += finish - start;

    start = clock();
    pc_par_ans = ParallelPredictorCorrectorMethod(y0, t0, tm, f, m);
    finish = clock();
    pc_par_time += finish - start;
  }
  std::cout << "Коэффициент ускорения метода Пикара: " << seq_time / par_time << std::endl;
  std::cout << "Коэффициент ускорения метода Рунге-Кутты: " << rk_seq_time / rk_par_time << std::endl;
  std::cout << "Сравнение эффективности метода Рунге-Кутты с методом Пикара" << std::endl;
  std::cout << "Последовательные реализации: " << seq_time / rk_seq_time << std::endl;
  std::cout << "Параллельные реализации: " << par_time / rk_par_time << std::endl;
  std::cout << "Коэффициент ускорения метода Адамса: " << pc_seq_time / pc_par_time << std::endl;

  // Рассчитаем погрешности полученных ответов

  // Инициальзируем вектор функторов аналитического решения системы
  std::vector<FunctorAns> ans{{{1, 1, 1, 0}, {-2, 2, 0, 0}, {2, 2}},
                              {{1, 1, 1, 0}, {1, 1, 1, 0}, {2, 2}},
                              {{1, 1, 1, 0}, {3, 3, -1, 0}, {2, 2}},
                              {{0, 0, 2, -1}, {-2, -1, 1, 1}, {1, 1}},
                              {{0, 0, 2, -1}, {1, 3, 2, -1}, {1, 1}}};
  const double h = (tm - t0) / m;
  double seq_eps = 0, par_eps = 0, rk_seq_eps = 0, rk_par_eps = 0, pc_seq_eps = 0, pc_par_eps = 0;
  double t = t0;
  for (int i = 0; i <= m; ++i, t += h) {
    for (int j = 0; j < n; ++j) {
      // Значение аналитического решения в точке t
      double ans_j = ans[j](t);
      seq_eps += fabs(seq_ans[i][j] - ans_j);
      par_eps += fabs(par_ans[i][j] - ans_j);
      rk_seq_eps += fabs(rk_seq_ans[i][j] - ans_j);
      rk_par_eps += fabs(rk_par_ans[i][j] - ans_j);
      pc_seq_eps += fabs(pc_seq_ans[i][j] - ans_j);
      pc_par_eps += fabs(pc_par_ans[i][j] - ans_j);
      //std::cout << ans_j << std::endl;
      //std::cout << seq_ans[i][j] << std::endl;
      //std::cout << par_ans[i][j] << std::endl;
      //std::cout << rk_ans[i][j] << std::endl;
      //std::cout << std::endl;
    }
  }
  std::cout << "Погрешность последовательного решения методом Рунге-Кутты: " << rk_seq_eps << std::endl;
  std::cout << "Погрешность параллельного решения методом Рунге-Кутты: " << rk_par_eps << std::endl;
  std::cout << "Погрешность последовательного решения методом Адамса: " << pc_seq_eps << std::endl;
  std::cout << "Погрешность параллельного решения методом Адамса: " << pc_par_eps << std::endl;

  return 0;
}