#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <bits/c++config.h>
#include <densematgen.h>
// #include <fmt/ostream.h>
#include <mpi.h>
#include <range.h>
#include <utils.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace matrix {

class MatrixSlice {
   public:
    using field_t = int32_t;
    using data_t = double;
    using metadata_t = std::vector<field_t>;
    const static field_t num_metadata_fields = 6;

    MatrixSlice(field_t num_rows,
                field_t num_cols,
                field_t rows_beg_idx,
                field_t rows_end_idx,
                field_t cols_beg_idx,
                field_t cols_end_idx);

    MatrixSlice(const metadata_t& metadata);

    inline field_t global_num_rows() const noexcept { return this->num_rows_; }
    inline field_t global_num_cols() const noexcept { return this->num_cols_; }
    inline field_t global_num_elems() const noexcept {
        return this->global_num_rows() * this->global_num_cols();
    }

    inline field_t local_num_rows() const noexcept {
        return this->rows_end_idx_ - this->rows_beg_idx_;
    }
    inline field_t local_num_cols() const noexcept {
        return this->cols_end_idx_ - this->cols_beg_idx_;
    }
    inline field_t local_num_elems() const noexcept {
        return this->local_num_rows() * this->local_num_cols();
    }

    inline field_t global_to_local_row(field_t global_row) const noexcept {
        return global_row - this->rows_beg_idx_;
    }
    inline field_t local_to_global_row(field_t local_row) const noexcept {
        return this->rows_beg_idx_ + local_row;
    }

    inline field_t global_to_local_col(field_t global_col) const noexcept {
        return global_col - this->cols_beg_idx_;
    }
    inline field_t local_to_global_col(field_t local_col) const noexcept {
        return local_col + this->cols_beg_idx_;
    }

    field_t local_non_zero_elems() const noexcept;

    virtual const std::vector<data_t>& data() const noexcept = 0;

    virtual std::vector<data_t>& data_mut() noexcept = 0;

    virtual void set_metadata(const metadata_t& metadata);

    virtual metadata_t get_metadata() const;

    friend std::ostream& operator<<(std::ostream& os, const MatrixSlice& mat);

   protected:
    field_t num_rows_;      // overall number of rows
    field_t num_cols_;      // overall number of columns
    field_t rows_beg_idx_;  // first row stored in this chunk
    field_t rows_end_idx_;  // first row not stored in this chunk
    field_t cols_beg_idx_;  // first column stored in this chunk
    field_t cols_end_idx_;  // first column not stored in this chunk
};

class Sparse;

class Dense : public MatrixSlice {
    // creates matrix filled with values generated with `generate_data_type`
    std::vector<data_t> data_;

   public:
    const static field_t num_metadata_fields = MatrixSlice::num_metadata_fields + 1;
    const static field_t data_size_metadata_idx = num_metadata_fields - 1;

    Dense(Dense&&) = default;

    Dense(const Dense&) = default;

    // creates matrix filled with zeros
    Dense(field_t num_rows,
          field_t num_cols,
          field_t rows_beg_idx,
          field_t rows_end_idx,
          field_t cols_beg_idx,
          field_t cols_end_idx);

    Dense(field_t num_rows,
          field_t num_cols,
          field_t rows_beg_idx,
          field_t rows_end_idx,
          field_t cols_beg_idx,
          field_t cols_end_idx,
          int seed);

    Dense(field_t num_rows,
          field_t num_cols,
          field_t rows_beg_idx,
          field_t rows_end_idx,
          field_t cols_beg_idx,
          field_t cols_end_idx,
          std::vector<data_t>&& data);

    Dense(const metadata_t& metadata, std::vector<data_t>&& data);

    /**
     * Creates dense matrix of the same shape as other but filled with zeros
     * @param other template matrix
     * @return matrix filled with zeros
     */
    static Dense like(const Dense& other) noexcept;

    void zero_data();

    virtual const std::vector<data_t>& data() const noexcept override;

    virtual std::vector<data_t>& data_mut() noexcept override { return this->data_; }

    inline double data_at_local(field_t row, field_t col) const {
        return this->data_[row * this->local_num_cols() + col];
    }

    inline double& data_at_local_mut(field_t row, field_t col) {
        return this->data_[row * this->local_num_cols() + col];
    }

    inline double data_at_global(field_t row, field_t col) const {
        return this->data_at_local(this->global_to_local_row(row), this->global_to_local_col(col));
    }

    inline double& data_at_global_mut(field_t row, field_t col) {
        return this->data_[this->global_to_local_row(row) * this->local_num_cols() +
                           this->global_to_local_col(col)];
    }

    virtual void set_metadata(const metadata_t& metadata) override final;

    virtual metadata_t get_metadata() const override final;

    bool same_shape_as(const Dense& other) const noexcept;

    utils::detail::Range<int32_t> rows_range() const;

    utils::detail::Range<int32_t> cols_range() const;

    utils::Span<data_t> data_at_row(field_t row) const;

    friend std::ostream& operator<<(std::ostream& os, const Dense& dt);

    friend std::ostream& operator<<(std::ostream& os, const std::vector<Dense>& mats);

    friend void matmul_sparse_dense(const Sparse& a, const Dense& b, Dense& target);
};

struct RawDenseParts {
    Dense::field_t num_rows;
    Dense::field_t num_parts;
    std::vector<Dense::data_t> parts;
    friend std::ostream& operator<<(std::ostream& os, const RawDenseParts& parts);
};

class Sparse : public MatrixSlice {
    std::vector<field_t> rows_offsets_;
    std::vector<field_t> cols_indices_;
    std::vector<data_t> data_;

   public:
    const static field_t num_metadata_fields = MatrixSlice::num_metadata_fields + 3;
    const static field_t rows_offsets_size_metadata_idx = num_metadata_fields - 3;
    const static field_t cols_indices_size_metadata_idx = num_metadata_fields - 2;
    const static field_t data_size_metadata_idx = num_metadata_fields - 1;

    explicit Sparse();

    Sparse(field_t num_rows,
           field_t num_cols,
           field_t rows_beg_idx,
           field_t rows_end_idx,
           field_t cols_beg_idx,
           field_t cols_end_idx,
           std::vector<field_t>&& rows_offsets,
           std::vector<field_t>&& cols_indices,
           std::vector<data_t>&& data);

    Sparse(const metadata_t& metadata,
           std::vector<field_t>&& rows_offsets,
           std::vector<field_t>&& cols_indices,
           std::vector<data_t>&& data);

    Sparse(const Sparse&) = delete;

    Sparse(Sparse&&) = default;

    Sparse& operator=(Sparse&&) = default;

    /**
     * Constructs Sparse matrix from parts.
     * Assumes parts are sorted.
     * @param parts vector of Sparse matrices
     * @param columnwise whether matrices are partitioned columnwise
     */
    explicit Sparse(const std::vector<Sparse>& parts, bool expand_local_num_rows);

    [[nodiscard]] std::vector<Sparse> into_parts(field_t num_parts,
                                                 bool shrink_local_num_rows) const;

    inline virtual const std::vector<data_t>& data() const noexcept final;

    virtual std::vector<data_t>& data_mut() noexcept override final;

    const std::vector<field_t>& rows_offsets() const noexcept;

    const std::vector<field_t>& cols_indices() const noexcept;

    virtual void set_metadata(const metadata_t& metadata) override final;

    virtual metadata_t get_metadata() const override final;

    utils::detail::Range<int32_t> rows_range() const;

    utils::Span<field_t> cols_at_row(field_t row) const;

    utils::Span<data_t> data_at_row(field_t row) const;

    friend std::ostream& operator<<(std::ostream& os, const Sparse& dt);
    friend void matmul_sparse_dense(const Sparse& a, const Dense& b, Dense& target);
};

}  // namespace matrix

#define MATRIX_DATA_T MPI_DOUBLE
#define MATRIX_FIELD_T MPI_INT32_T

#endif  // __MATRIX_H__
