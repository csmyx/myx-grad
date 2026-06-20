#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

namespace util {
class Arena {
public:
    Arena() = default;
    ~Arena();
    Arena(const Arena &) = delete;
    auto operator=(const Arena &) -> Arena & = delete;
    Arena(Arena &&) = default;
    auto operator=(Arena &&) -> Arena & = default;

    inline auto alloc(size_t size) -> void *;
    inline auto alloc_aligned(size_t size) -> void *;

private:
    static constexpr size_t chunk_size = 4096;
    static constexpr size_t big_alloc_size = chunk_size / 4;
    char *cur_ptr_ = nullptr;
    size_t remaining_size_ = 0;
    std::vector<char *> chunks_;
};

inline auto Arena::alloc(size_t size) -> void * {
    if (size > big_alloc_size) {
        // Allocate seperately for big object
        char *big_chunk = new char[size];
        chunks_.push_back(big_chunk);
        return big_chunk;
    }

    if (remaining_size_ < size) {
        // Allocate a new chunk, discard the remaining space in the old chunk,
        // It's fine because the wasted size is not greater than `big_alloc_size`.
        char *new_chunk = new char[chunk_size];
        chunks_.push_back(new_chunk);
        cur_ptr_ = new_chunk;
        remaining_size_ = chunk_size;
    }
    assert(cur_ptr_ != nullptr && remaining_size_ >= size);

    char *ptr = cur_ptr_;
    cur_ptr_ += size;
    remaining_size_ -= size;
    return ptr;
}

inline auto Arena::alloc_aligned(size_t size) -> void * {
    if (size > big_alloc_size) {
        // Allocate seperately for big object
        char *big_chunk = new char[size];
        chunks_.push_back(big_chunk);
        return big_chunk;
    }

    if (remaining_size_ < size) {
        // Allocate a new chunk, discard the remaining space in the old chunk,
        // It's fine because the wasted size is not greater than `big_alloc_size`.
        char *new_chunk = new char[chunk_size];
        chunks_.push_back(new_chunk);
        cur_ptr_ = new_chunk;
        remaining_size_ = chunk_size;
    }
    assert(cur_ptr_ != nullptr && remaining_size_ >= size);

    char *ptr = cur_ptr_;
    cur_ptr_ += size;
    remaining_size_ -= size;
    return ptr;
}

}  // namespace util