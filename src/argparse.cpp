#include <argparse.h>

Args::Args(int argc, char** argv) : argc(argc), argv(argv) {
    std::unordered_set<std::string> binary_flags{"-f", "-s", "-c", "-e", "-g"};
    std::unordered_set<std::string> unary_flags{"-v", "-i"};

    int idx = 1;
    while (idx < argc) {
        std::string argument = argv[idx];
        if (argument == "-f") {
            this->sparse_matrix_file = argv[idx + 1];
        } else if (argument == "-s") {
            this->seed_for_dense_matrix = std::stoi(argv[idx + 1]);
        } else if (argument == "-c") {
            this->repl_group_size = std::stoi(argv[idx + 1]);
        } else if (argument == "-e") {
            this->exponent = std::stoi(argv[idx + 1]);
        } else if (argument == "-g") {
            this->ge_value = std::stof(argv[idx + 1]);
        } else if (argument == "-v") {
            this->verbose = true;
        } else if (argument == "-i") {
            this->inner_algorithm = true;
        } else {
            throw std::invalid_argument(std::string("Unknown option: ") + argv[idx]);
        }

        if (binary_flags.find(argument) != binary_flags.end()) {
            idx += 2;
        }

        if (unary_flags.find(argument) != unary_flags.end()) {
            idx += 1;
        }
    }

    if (this->sparse_matrix_file.empty() || this->seed_for_dense_matrix == -1 ||
        this->repl_group_size == -1 || this->exponent == -1) {
        throw std::invalid_argument(
            std::string("Usage: ") + argv[0] +
            " -f sparse_matrix_file -s seed_for_dense_matrix -c repl_group_size -e "
            "exponent [-g ge_value] [-v] [-i]");
    }
}
