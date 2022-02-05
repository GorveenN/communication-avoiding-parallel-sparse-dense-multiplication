#ifndef __COMMUNICATOR_H__
#define __COMMUNICATOR_H__

#include <matrix.h>
#include <mpi.h>
#include <memory>
#include <optional>

// TODO: add async version of operations

class Communicator {
   private:
    MPI_Comm _comm;
    int32_t _world_size;
    int32_t _rank;
    std::vector<MPI_Request> _requests;

   public:
    using field_t = matrix::MatrixSlice::field_t;
    using data_t = matrix::MatrixSlice::data_t;
    using metadata_t = matrix::MatrixSlice::metadata_t;
    Communicator(int argc, char* argv[]);
    Communicator(const Communicator& comm, int replication_group_size, bool planewise);
    Communicator(const Communicator& comm, int color);

    virtual ~Communicator();

    void send_dense(const matrix::Dense& mat, int target);
    matrix::Dense recv_dense(int sender);
    void bcast_send_dense(const matrix::Sparse& mat);
    matrix::Dense bcast_recv_dense(int sender);
    std::optional<matrix::RawDenseParts> gather_dense_parts(const matrix::Dense& mat,
                                             int target,
                                             const std::vector<int>& displs,
                                             const std::vector<int>& recvcounts);
    void gather_dense_into(matrix::Dense& mat,
                           std::vector<int> displs,
                           std::vector<int> recvcounts);

    std::vector<std::vector<field_t>> gather_sparse_metadata(const matrix::Sparse& mat);
    void isend_sparse(const matrix::Sparse& mat, int target);
    void irecv_sparse(matrix::Sparse& mat, int sender);

    void send_sparse(const matrix::Sparse& mat, int target);
    matrix::Sparse recv_sparse(int sender);
    void bcast_send_sparse(const matrix::Sparse& mat);
    matrix::Sparse bcast_recv_sparse(int sender);

    std::optional<int> reduce_int(int i, int root = 0);

    void wait_all();
    int32_t world_size();
    int32_t rank();
    bool is_coordinator();
};

#endif  // __COMMUNICATOR_H__
