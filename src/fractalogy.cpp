#include <iostream>
#include <complex>

using std::cout;
using std::complex;
using std::exp;
using std::norm;

/// returns -1 on failing to escape
int iteration(complex<double> c, int limit = 1000) {
  int i = 0;
  double n;
  complex<double> z(0, 0);
  while ((n = norm(z)) < 4 && i < limit) {
    z = exp(z) - c;
    ++i;
  }

  if (n < 4)
    return -1;
  else
    return i;
}

int main() {
  std::cout << "Fractalogy!\n";
  double lower = -2, upper = 2;
  int width = 100, height = 100;
  for (int i = 0; i <= width; ++i) {
    for (int j = 0; j <= height; ++j) {
      complex<double> t;
      t = complex<double>(lower + (upper - lower) * i / width, lower + (upper - lower) * j / height);
      cout << iteration(t) << " ";
    }
    cout << std::endl;
  }

  return 0;
}

