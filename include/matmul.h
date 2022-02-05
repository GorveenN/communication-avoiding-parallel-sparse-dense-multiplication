#ifndef __MATMUL_H__
#define __MATMUL_H__

#include <argparse.h>
#include <communicator.h>
#include <matrix.h>

namespace matmul {

class MatmulAlg {
   protected:
    Args args;
    std::unique_ptr<matrix::Dense> dense_;
    std::unique_ptr<matrix::Dense> result_;
    std::unique_ptr<matrix::Sparse> sparse_;
    std::unique_ptr<matrix::Sparse> spare_sparse_;
    Communicator global_comm_;
    Communicator repl_group_comm_;
    Communicator repl_plane_comm_;
    std::vector<std::vector<matrix::Sparse::field_t>> metadata_;

    virtual std::tuple<int, int, int> sparse_shift_indices(int step) = 0;
    virtual int dense_plane_num_chunks() = 0;
    virtual int dense_plane_chunk_idx() = 0;
    virtual bool sparse_partitioned_rowwise() = 0;
    virtual bool should_gather_dense() = 0;
    virtual int dense_num_chunks() = 0;

    void gather_sparse_metadata();
    void prepare_data();
    void prepare_matrices();
    void read_and_distribute_sparse();
    void replicate_sparse();
    void generate_dense();
    void generate_result_matrix();

    void schedule_shift_sparse(int step);
    void finish_shift_sparse();

    bool is_coordinator();

   public:
    explicit MatmulAlg(const Args& args);
    virtual ~MatmulAlg() = default;

    virtual void run() = 0;
    virtual std::optional<matrix::RawDenseParts> collect_denses();
    virtual std::optional<int> result_num_greater_than(double ge);
};

class InnerABC : public MatmulAlg {
    // dense:
    // 0 | 1 | 2 | 3
    // 4 | 5 | 6 | 7

    // sparse:
    // 0 1
    // ---
    // 2 3
    // ---
    // 4 5
    // ---
    // 6 7

   private:
    Communicator repl_plane_group_comm_;

    void sync_dense();
    std::pair<std::vector<int>, std::vector<int>> dense_shift_displ_recv();

   protected:
    virtual bool should_gather_dense() override;
    virtual int dense_num_chunks() override;
    virtual std::tuple<int, int, int> sparse_shift_indices(int step) override;
    virtual int dense_plane_num_chunks() override;
    virtual int dense_plane_chunk_idx() override;
    virtual bool sparse_partitioned_rowwise() override;

   public:
    explicit InnerABC(const Args& args);
    virtual ~InnerABC() = default;
    void run() override;
};

class ColA : public MatmulAlg {
   protected:
    virtual bool should_gather_dense() override;
    virtual int dense_num_chunks() override;
    virtual std::tuple<int, int, int> sparse_shift_indices(int step) override;
    virtual int dense_plane_num_chunks() override;
    virtual int dense_plane_chunk_idx() override;
    virtual bool sparse_partitioned_rowwise() override;

   public:
    explicit ColA(const Args& args);
    virtual ~ColA() = default;
    virtual void run() override;
};

};  // namespace matmul

#endif  // __MATMUL_H__
