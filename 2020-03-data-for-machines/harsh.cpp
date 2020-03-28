#include <iostream>

int main() {
  int a = 42;
  int b = 30;
  int l = 10;
  auto z = [](int x) { return x; };
  int qq = 2;
  int q = 1;

  auto y = (a?~b:-b)*(l+1/z(2))>>qq/q;

  std::cout << y << std::endl;
  return 0;
}
