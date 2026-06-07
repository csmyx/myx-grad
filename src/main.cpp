#include "myx_grad.h"
#include <fmt/core.h>
#include <string>
#include <vector>

int main() {
  myx_grad();

  std::vector<std::string> vec;
  vec.push_back("test_package");
  myx_grad_print_vector(vec);

  fmt::println("hello");
}
