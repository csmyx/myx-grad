#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
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

    inline auto alloc(size_t size, size_t align) -> void *;
    inline auto alloc_unaligned(size_t size) -> void *;

    template <typename T, size_t size = sizeof(T), size_t align = alignof(T)>
    inline auto alloc() -> void *;

private:
    static constexpr size_t chunk_size = 4096;
    static constexpr size_t big_alloc_size = chunk_size / 4;
    char *cur_ptr_ = nullptr;
    size_t remaining_size_ = 0;
    std::vector<char *> chunks_;
    inline auto alloc_fallback(size_t size) -> char *;
};


inline auto Arena::alloc(size_t size, size_t align) -> void * {
    assert(size > 0 && align > 0);
    if (size > big_alloc_size) {
        // Allocate seperately for big object
        char *big_alloc = new char[size];
        return big_alloc;
    }

    const auto current_addr = reinterpret_cast<uintptr_t>(cur_ptr_);
    const uintptr_t aligned_addr = (current_addr + align - 1) & ~(align - 1);
    const size_t padding = aligned_addr - current_addr;
    const size_t size_aligned = padding + size;

    char *ptr = alloc_fallback(size_aligned);
    return ptr + padding;
}

inline auto Arena::alloc_unaligned(size_t size) -> void * {
    assert(size > 0);
    if (size > big_alloc_size) {
        // Allocate seperately for big object
        char *big_alloc = new char[size];
        return big_alloc;
    }
    char *ptr = alloc_fallback(size);
    return ptr;
}

inline auto Arena::alloc_fallback(size_t size) -> char * {
    assert(size > 0);
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

template <typename T, size_t size, size_t align>
inline auto Arena::alloc() -> void * {
    return alloc(size, align);
}
}  // namespace util