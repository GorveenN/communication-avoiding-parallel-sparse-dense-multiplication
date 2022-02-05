#include <parser.h>
#include <fstream>

matrix::Sparse Parser::parse(std::ifstream& stream) {
    int32_t num_rows, num_cols, num_non_zero, max_non_zero_per_row;
    stream >> num_rows >> num_cols >> num_non_zero >> max_non_zero_per_row;

    std::vector<int32_t> rows(num_rows + 1);
    std::vector<int32_t> cols(num_non_zero);
    std::vector<double> data(num_non_zero);

    for (double& datum : data) {
        stream >> datum;
    }

    for (int& row : rows) {
        stream >> row;
    }

    for (int& col : cols) {
        stream >> col;
    }

    return matrix::Sparse(num_rows, num_cols, 0, num_rows, 0, num_cols, std::move(rows),
                          std::move(cols), std::move(data));
}
