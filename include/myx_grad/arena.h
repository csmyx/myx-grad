#pragma once

#include <cstddef>
#include <vector>

class Arena {
public:
    Arena() {}
private:
    static constexpr size_t chunk_size = 4096;
    char *cur_chunk_;
    std::vector<char *> chunks_;
};