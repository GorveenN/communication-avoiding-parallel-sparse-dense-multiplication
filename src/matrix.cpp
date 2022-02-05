#include <matrix.h>
#include <utils.h>
#include <zip.h>

namespace matrix {
using data_t = MatrixSlice::data_t;
using field_t = MatrixSlice::field_t;

// MatrixSlice
MatrixSlice::MatrixSlice(field_t num_rows,
                         field_t num_cols,
                         field_t rows_beg_idx,
                         field_t rows_end_idx,
                         field_t cols_beg_idx,
                         field_t cols_end_idx)
    : num_rows_{num_rows},
      num_cols_{num_cols},
      rows_beg_idx_{rows_beg_idx},
      rows_end_idx_{rows_end_idx},
      cols_beg_idx_{cols_beg_idx},
      cols_end_idx_{cols_end_idx} {}

MatrixSlice::MatrixSlice(const metadata_t& metadata)
    : num_rows_{metadata[0]},
      num_cols_{metadata[1]},
      rows_beg_idx_{metadata[2]},
      rows_end_idx_{metadata[3]},
      cols_beg_idx_{metadata[4]},
      cols_end_idx_{metadata[5]} {}

field_t MatrixSlice::local_non_zero_elems() const noexcept {
    return this->data().size();
}

void MatrixSlice::set_metadata(const MatrixSlice::metadata_t& metadata) {
    this->num_rows_ = metadata[0];
    this->num_cols_ = metadata[1];
    this->rows_beg_idx_ = metadata[2];
    this->rows_end_idx_ = metadata[3];
    this->cols_beg_idx_ = metadata[4];
    this->cols_end_idx_ = metadata[5];
}

MatrixSlice::metadata_t MatrixSlice::get_metadata() const {
    return {this->num_rows_,     this->num_cols_,     this->rows_beg_idx_,
            this->rows_end_idx_, this->cols_beg_idx_, this->cols_end_idx_};
}

std::ostream& operator<<(std::ostream& os, const MatrixSlice& mat) {
    os << mat.num_rows_ << " " << mat.num_cols_ << " " << mat.rows_beg_idx_ << " "
       << mat.rows_end_idx_ << " " << mat.cols_beg_idx_ << " " << mat.cols_end_idx_ << " "
       << std::endl;
    return os;
}

// Dense
Dense::Dense(field_t num_rows,
             field_t num_cols,
             field_t rows_beg_idx,
             field_t rows_end_idx,
             field_t cols_beg_idx,
             field_t cols_end_idx)
    : MatrixSlice(num_rows, num_cols, rows_beg_idx, rows_end_idx, cols_beg_idx, cols_end_idx),
      data_(num_rows * (cols_end_idx - cols_beg_idx), 0) {}

Dense::Dense(field_t num_rows,
             field_t num_cols,
             field_t rows_beg_idx,
             field_t rows_end_idx,
             field_t cols_beg_idx,
             field_t cols_end_idx,
             int seed)
    : MatrixSlice(num_rows, num_cols, rows_beg_idx, rows_end_idx, cols_beg_idx, cols_end_idx),
      data_(num_rows * (cols_end_idx - cols_beg_idx)) {
    for (const auto row : utils::range(this->local_num_rows())) {
        for (const auto col : utils::range(this->local_num_cols())) {
            this->data_at_local_mut(row, col) = generate_double(
                seed, this->global_to_local_row(row), this->local_to_global_col(col));
        }
    }
}

Dense::Dense(field_t num_rows,
             field_t num_cols,
             field_t rows_beg_idx,
             field_t rows_end_idx,
             field_t cols_beg_idx,
             field_t cols_end_idx,
             std::vector<data_t>&& data)
    : MatrixSlice(num_rows, num_cols, rows_beg_idx, rows_end_idx, cols_beg_idx, cols_end_idx),
      data_(std::move(data)) {}

Dense::Dense(const metadata_t& metadata, std::vector<data_t>&& data)
    : MatrixSlice{metadata}, data_{std::move(data)} {}

Dense Dense::like(const Dense& other) noexcept {
    return Dense{other.num_rows_,     other.num_cols_,     other.rows_beg_idx_,
                 other.rows_end_idx_, other.cols_beg_idx_, other.cols_end_idx_};
}

void Dense::zero_data() {
    std::fill(this->data_.begin(), this->data_.end(), 0.0);
}

const std::vector<data_t>& Dense::data() const noexcept {
    return this->data_;
}

void Dense::set_metadata(const metadata_t& metadata) {
    MatrixSlice::set_metadata(metadata);
    this->data_.resize(metadata[data_size_metadata_idx]);
}

Dense::metadata_t Dense::get_metadata() const {
    auto metadata = MatrixSlice::get_metadata();
    metadata.emplace_back(this->data_.size());
    return metadata;
}

bool Dense::same_shape_as(const Dense& other) const noexcept {
    return this->num_rows_ == other.num_rows_ && this->num_cols_ == other.num_cols_ &&
           this->rows_beg_idx_ == other.rows_beg_idx_ &&
           this->rows_end_idx_ == other.rows_end_idx_ &&
           this->cols_beg_idx_ == other.cols_beg_idx_ && this->cols_end_idx_ == other.cols_end_idx_;
}

utils::detail::Range<int32_t> Dense::rows_range() const {
    return utils::range(this->local_num_rows());
}

utils::detail::Range<int32_t> Dense::cols_range() const {
    return utils::range(this->local_num_cols());
}

utils::Span<data_t> Dense::data_at_row(field_t row) const {
    return utils::Span<data_t>{this->data_.data() + row * local_num_cols(),
                               static_cast<size_t>(this->local_num_cols())};
}

std::ostream& operator<<(std::ostream& os, const std::vector<Dense>& mats) {
    if (mats.empty()) {
        return os;
    }

    for (const auto& row : mats[0].rows_range()) {
        for (const auto& mat : mats) {
            for (const auto& datum : mat.data_at_row(row)) {
                os << datum << "  ";
            }
        }
        os << std::endl;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const Dense& mat) {
    auto num_rows = mat.local_num_rows();
    auto num_cols = mat.local_num_cols();

    os << mat.num_rows_ << " " << mat.num_cols_ << " " << mat.rows_beg_idx_ << " "
       << mat.rows_end_idx_ << " " << mat.cols_beg_idx_ << " " << mat.cols_end_idx_ << " "
       << std::endl;

    for (const auto& row : utils::range(num_rows)) {
        for (const auto& col : utils::range(num_cols)) {
            os << mat.data_at_local(row, col);
            if (col < num_cols - 1) {
                os << " ";
            }
        }
        if (row < num_rows - 1) {
            os << std::endl;
        }
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const RawDenseParts& parts) {
    auto [offsets, sizes] = utils::partition_number(parts.num_rows, parts.num_parts);

    os << parts.num_rows << " " << parts.num_rows << std::endl;

    for (const auto& row : utils::range(parts.num_rows)) {
        for (const auto& [mat_offset, mat_size] : utils::zip(offsets, sizes)) {
            for (const auto& datum : utils::Span(
                     parts.parts.data() + mat_offset * parts.num_rows + row * mat_size, mat_size)) {
                std::cout << datum << " ";
            }
        }
        os << std::endl;
    }

    return os;
}

Sparse::Sparse() : MatrixSlice(0, 0, 0, 0, 0, 0) {}

Sparse::Sparse(field_t num_rows,
               field_t num_cols,
               field_t rows_beg_idx,
               field_t rows_end_idx,
               field_t cols_beg_idx,
               field_t cols_end_idx,
               std::vector<field_t>&& rows_offsets,
               std::vector<field_t>&& cols_indices,
               std::vector<data_t>&& data)
    : MatrixSlice(num_rows, num_cols, rows_beg_idx, rows_end_idx, cols_beg_idx, cols_end_idx),
      rows_offsets_(std::move(rows_offsets)),
      cols_indices_(std::move(cols_indices)),
      data_(std::move(data)) {}

Sparse::Sparse(const metadata_t& metadata,
               std::vector<field_t>&& rows_offsets,
               std::vector<field_t>&& cols_indices,
               std::vector<data_t>&& data)
    : MatrixSlice{metadata},
      rows_offsets_{std::move(rows_offsets)},
      cols_indices_{cols_indices},
      data_{std::move(data)} {}

Sparse::Sparse(const std::vector<Sparse>& parts, bool expand_local_num_rows)
    : MatrixSlice(parts[0].num_rows_,
                  parts[0].num_cols_,
                  parts[0].rows_beg_idx_,
                  parts[parts.size() - 1].rows_end_idx_,
                  parts[0].cols_beg_idx_,
                  parts[parts.size() - 1].cols_end_idx_) {
    this->rows_offsets_.emplace_back(0);
    if (expand_local_num_rows) {
        for (const auto& part : parts) {
            for (const auto& row : part.rows_range()) {
                for (const auto& [col, data] :
                     utils::zip(part.cols_at_row(row), part.data_at_row(row))) {
                    this->cols_indices_.emplace_back(
                        this->global_to_local_col(part.local_to_global_col(col)));
                    this->data_.emplace_back(data);
                }
                this->rows_offsets_.emplace_back(this->data_.size());
            }
        }
    } else {
        for (const auto& row : this->rows_range()) {
            for (const auto& part : parts) {
                for (const auto& [col, data] :
                     utils::zip(part.cols_at_row(row), part.data_at_row(row))) {
                    this->cols_indices_.emplace_back(
                        this->global_to_local_col(part.local_to_global_col(col)));
                    this->data_.emplace_back(data);
                }
            }
            this->rows_offsets_.emplace_back(this->cols_indices_.size());
        }
    }
}

std::vector<Sparse> Sparse::into_parts(const int32_t num_parts, bool shrink_local_num_rows) const {
    if (shrink_local_num_rows) {
        auto [parts_offsets, parts_sizes] =
            utils::partition_number(this->local_num_cols(), num_parts);

        std::vector<matrix::Sparse> parts;
        for (const auto& [rows_offsets_part_beg_idx, cur_part_num_rows] :
             utils::zip(parts_offsets, parts_sizes)) {
            auto rows_offsets_beg_iter = rows_offsets_.begin() + rows_offsets_part_beg_idx;
            auto rows_offsets_end_iter = rows_offsets_beg_iter + cur_part_num_rows + 1;

            std::vector<field_t> part_rows_offsets{rows_offsets_beg_iter, rows_offsets_end_iter};
            std::vector<field_t> part_cols_indices{
                this->cols_indices_.begin() + *rows_offsets_beg_iter,
                this->cols_indices_.begin() + *(rows_offsets_end_iter - 1)};
            std::vector<data_t> part_data{this->data().begin() + *rows_offsets_beg_iter,
                                          this->data().begin() + *(rows_offsets_end_iter - 1)};

            auto base_offset = part_rows_offsets[0];
            for (auto& part_rows_offset : part_rows_offsets) {
                part_rows_offset -= base_offset;
            }

            parts.emplace_back(Sparse{this->num_rows_, this->num_cols_, rows_offsets_part_beg_idx,
                                      rows_offsets_part_beg_idx + cur_part_num_rows,
                                      this->cols_beg_idx_, this->cols_end_idx_,
                                      std::move(part_rows_offsets), std::move(part_cols_indices),
                                      std::move(part_data)});
        }

        return parts;
    } else {
        auto [parts_offsets, parts_sizes] =
            utils::partition_number(this->local_num_cols(), num_parts);
        std::vector<matrix::Sparse> parts;
        for (const auto& [offset, size] : utils::zip(parts_offsets, parts_sizes)) {
            parts.emplace_back(Sparse(this->num_rows_, this->num_cols_, this->rows_beg_idx_,
                                      this->rows_end_idx_, offset, offset + size,
                                      std::vector<int32_t>{0}, std::vector<int32_t>(),
                                      std::vector<data_t>()));
        }

        for (auto row_idx : this->rows_range()) {
            for (const auto& [col_idx, data] :
                 utils::zip(this->cols_at_row(row_idx), this->data_at_row(row_idx))) {
                auto target_part_idx =
                    utils::target_idx_from_offset(this->local_num_cols(), num_parts, col_idx);
                auto& target_part = parts[target_part_idx];
                target_part.cols_indices_.emplace_back(
                    target_part.global_to_local_col(this->local_to_global_col(col_idx)));
                target_part.data_mut().emplace_back(data);
            }

            for (auto& part : parts) {
                part.rows_offsets_.emplace_back(part.cols_indices_.size());
            }
        }

        return parts;
    }
}

const std::vector<data_t>& Sparse::data() const noexcept {
    return this->data_;
}

std::vector<data_t>& Sparse::data_mut() noexcept {
    return this->data_;
}

const std::vector<field_t>& Sparse::rows_offsets() const noexcept {
    return this->rows_offsets_;
}
const std::vector<field_t>& Sparse::cols_indices() const noexcept {
    return this->cols_indices_;
}
Sparse::metadata_t Sparse::get_metadata() const {
    auto metadata = MatrixSlice::get_metadata();
    metadata.emplace_back(this->rows_offsets().size());
    metadata.emplace_back(this->cols_indices().size());
    metadata.emplace_back(this->data().size());
    return metadata;
}

void Sparse::set_metadata(const Sparse::metadata_t& metadata) {
    MatrixSlice::set_metadata(metadata);
    this->rows_offsets_.resize(metadata[rows_offsets_size_metadata_idx]);
    this->cols_indices_.resize(metadata[cols_indices_size_metadata_idx]);
    this->data_.resize(metadata[data_size_metadata_idx]);
}

utils::detail::Range<int32_t> Sparse::rows_range() const {
    return utils::range(this->local_num_rows());
}

utils::Span<field_t> Sparse::cols_at_row(field_t row) const {
    auto beg_idx = this->rows_offsets_[row];
    auto end_idx = this->rows_offsets_[row + 1];
    auto data = this->cols_indices_.data() + beg_idx;
    return utils::Span<field_t>{data, static_cast<size_t>(end_idx - beg_idx)};
}

utils::Span<data_t> Sparse::data_at_row(field_t row) const {
    auto beg_idx = this->rows_offsets_[row];
    auto end_idx = this->rows_offsets_[row + 1];
    auto data = this->data_.data() + beg_idx;
    return utils::Span<data_t>{data, static_cast<size_t>(end_idx - beg_idx)};
}

std::ostream& operator<<(std::ostream& os, const Sparse& mat) {
    os << mat.num_rows_ << " " << mat.num_cols_ << " " << mat.rows_beg_idx_ << " "
       << mat.rows_end_idx_ << " " << mat.cols_beg_idx_ << " " << mat.cols_end_idx_ << " "
       << std::endl;

    for (auto const& i : mat.rows_offsets_) {
        os << i << " ";
    }
    os << std::endl;

    for (auto const& i : mat.cols_indices_) {
        os << i << " ";
    }
    os << std::endl;

    for (auto const& i : mat.data()) {
        os << i << " ";
    }

    return os;
}

void matmul_sparse_dense(const Sparse& sparse, const Dense& dense, Dense& result) {
    for (auto s_local_row = 0; s_local_row < sparse.rows_end_idx_ - sparse.rows_beg_idx_;
         s_local_row++) {
        for (auto idx = sparse.rows_offsets_[s_local_row];
             idx < sparse.rows_offsets_[s_local_row + 1]; idx++) {
            const auto s_local_col = sparse.cols_indices_[idx];
            const auto s_data = sparse.data_[idx];
            const auto d_local_row = sparse.cols_beg_idx_ + s_local_col;
            const auto d_local_num_cols = dense.local_num_cols();
            for (auto d_local_col = 0; d_local_col < d_local_num_cols; d_local_col++) {
                const auto d_data = dense.data_[d_local_row * d_local_num_cols + d_local_col];
                result.data_[(s_local_row + sparse.rows_beg_idx_ - result.rows_beg_idx_) *
                                 d_local_num_cols +
                             d_local_col] += s_data * d_data;
            }
        }
    }
}

};  // namespace matrix
