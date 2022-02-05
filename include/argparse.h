#ifndef __ARGPARSE_H__
#define __ARGPARSE_H__

#include <optional>
#include <string>

#include <stdexcept>
#include <string>
#include <unordered_set>

struct Args {
    int argc;
    char** argv;
    std::string sparse_matrix_file;                // -f sparse_matrix_file
    int seed_for_dense_matrix = -1;                // -s seed_for_dense_matrix
    int repl_group_size = -1;                    // -c repl_group_size
    int exponent = -1;                             // -e exponent
    std::optional<float> ge_value = std::nullopt;  // -g ge_value
    bool verbose = false;                          // -v
    bool inner_algorithm = false;                  // -i

    Args(int argc, char** argv);
};

#endif  // __ARGPARSE_H__
