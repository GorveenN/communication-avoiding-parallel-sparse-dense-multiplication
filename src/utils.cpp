#include <mpi.h>
#include <range.h>
#include <utils.h>
#include <zip.h>
#include <iostream>
#include <numeric>

namespace utils {

std::pair<std::vector<int>, std::vector<int>> partition_number(int number, int num_chunks) {
    int32_t chunk_size = number % num_chunks == 0 ? number / num_chunks : 1 + number / num_chunks;
    int32_t smaller_chunk_size = number % num_chunks == 0 ? chunk_size : chunk_size - 1;
    int32_t parts_left = number - num_chunks * smaller_chunk_size;
    std::vector<int> chunk_sizes(num_chunks, smaller_chunk_size);

    for (const auto& i : utils::range(parts_left)) {
        chunk_sizes[i]++;
    }

    auto chunk_offsets = std::vector<int32_t>(chunk_sizes.size());
    std::partial_sum(chunk_sizes.begin(), chunk_sizes.end(), chunk_offsets.begin(),
                     std::plus<int>());
    chunk_offsets.insert(chunk_offsets.begin(), 0);
    chunk_offsets.pop_back();

    return std::make_pair(chunk_offsets, chunk_sizes);
}

int target_idx_from_offset(int number, int num_chunks, int idx) {
    int32_t bigger_chunk_size =
        number % num_chunks == 0 ? number / num_chunks : 1 + number / num_chunks;
    int32_t smaller_chunk_size =
        number % num_chunks == 0 ? bigger_chunk_size : bigger_chunk_size - 1;
    int32_t parts_left = number - num_chunks * smaller_chunk_size;

    if (idx < parts_left * bigger_chunk_size) {
        return idx / bigger_chunk_size;
    } else {
        return parts_left + (idx - parts_left * bigger_chunk_size) / (smaller_chunk_size);
    }
}

std::pair<int, int> partition_number_nth(int number, int num_chunks, int n) {
    auto [offsets, part_sizes] = partition_number(number, num_chunks);
    return std::make_pair(offsets[n], part_sizes[n]);
}

}  // namespace utils
