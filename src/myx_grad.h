#pragma once

#include <vector>
#include <string>


#ifdef _WIN32
  #define MYX_GRAD_EXPORT __declspec(dllexport)
#else
  #define MYX_GRAD_EXPORT
#endif

MYX_GRAD_EXPORT void myx_grad();
MYX_GRAD_EXPORT void myx_grad_print_vector(const std::vector<std::string> &strings);
