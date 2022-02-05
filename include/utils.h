#ifndef __UTILS_H__
#define __UTILS_H__

#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace utils {

#define DUMP(a)                                           \
    do {                                                  \
        std::cout << #a " is value " << (a) << std::endl; \
        \                                                 \
    } while (false)
class NotImplementedException : public std::logic_error {
   public:
    NotImplementedException() : std::logic_error{"Function not yet implemented."} {}
};

std::pair<std::vector<int>, std::vector<int>> partition_number(int number, int num_chunks);

int target_idx_from_offset(int number, int num_chunks, int idx);

/**
 * Returns <offset, length> of the nth part of whole partitioned into num_chunks parts.
 * @param n nth chunk
 * @param number size of the thing to partition
 * @param num_chunks number of chunks to partition whole into
 * @return <offset, length>
 */
std::pair<int, int> partition_number_nth(int number, int num_chunks, int n);

void prepare_logger();

template <typename T>
class Span {
    // based on https://internalpointers.com/post/writing-custom-iterators-modern-cpp
    const T* data_;
    size_t length_;

   public:
    Span(const T* data, size_t length) : data_{data}, length_{length} {}

    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = const value_type*;
        using reference = const value_type&;

        Iterator(pointer ptr) : m_ptr(ptr) {}

        reference operator*() const { return *m_ptr; }
        pointer operator->() { return m_ptr; }

        Iterator& operator++() {
            m_ptr++;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
        friend bool operator!=(const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };

       private:
        pointer m_ptr;
    };
    Iterator begin() { return Iterator(&data_[0]); }
    Iterator end() { return Iterator(&data_[this->length_]); }

    T* data() { return this->data_; }
};

inline bool file_exists(const std::string& name) {
    // intel can't into c++17
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

};  // namespace utils

#endif  // __UTILS_H__
