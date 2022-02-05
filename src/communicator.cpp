#include <communicator.h>
#include <mpi.h>
#include <range.h>
#include <utils.h>
#include <zip.h>
#include <algorithm>
#include <numeric>

Communicator::Communicator(int argc, char* argv[]) : _comm(MPI_COMM_WORLD) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &this->_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->_rank);
}

Communicator::Communicator(const Communicator& comm, int repl_group_size, bool planewise) {
    int color = planewise ? comm._rank % repl_group_size : comm._rank / repl_group_size;
    MPI_Comm_split(comm._comm, color, comm._rank, &this->_comm);
    MPI_Comm_size(this->_comm, &this->_world_size);
    MPI_Comm_rank(this->_comm, &this->_rank);
}

Communicator::Communicator(const Communicator& comm, int color) {
    MPI_Comm_split(comm._comm, color, comm._rank, &this->_comm);
    MPI_Comm_size(this->_comm, &this->_world_size);
    MPI_Comm_rank(this->_comm, &this->_rank);
}

Communicator::~Communicator() {
    if (this->_comm == MPI_COMM_WORLD) {
        MPI_Finalize();
    } else {
        MPI_Comm_free(&this->_comm);
    }
}

void Communicator::send_dense(const matrix::Dense& mat, int target) {
    auto metadata = mat.get_metadata();
    MPI_Send(metadata.data(), metadata.size(), MATRIX_FIELD_T, target, 0, this->_comm);
    MPI_Send(mat.data().data(), mat.data().size(), MATRIX_DATA_T, target, 0, this->_comm);
}

matrix::Dense Communicator::recv_dense(int source) {
    metadata_t metadata(matrix::Dense::num_metadata_fields);
    MPI_Recv(metadata.data(), metadata.size(), MATRIX_FIELD_T, source, 0, this->_comm,
             MPI_STATUS_IGNORE);
    field_t data_size = metadata[matrix::Dense::data_size_metadata_idx];
    std::vector<data_t> data(data_size);
    MPI_Recv(data.data(), data.size(), MATRIX_DATA_T, source, 0, this->_comm, MPI_STATUS_IGNORE);
    return matrix::Dense(metadata, std::move(data));
}

void Communicator::bcast_send_dense(const matrix::Sparse& mat) {
    auto metadata = mat.get_metadata();
    MPI_Bcast(metadata.data(), metadata.size(), MATRIX_FIELD_T, this->_rank, this->_comm);
    MPI_Bcast((void*)mat.data().data(), mat.data().size(), MATRIX_DATA_T, this->_rank, this->_comm);
}

matrix::Dense Communicator::bcast_recv_dense(int sender) {
    metadata_t metadata(matrix::Dense::num_metadata_fields);
    MPI_Bcast(metadata.data(), metadata.size(), MATRIX_FIELD_T, sender, this->_comm);
    std::vector<data_t> data(metadata[matrix::Dense::data_size_metadata_idx]);
    MPI_Bcast(data.data(), data.size(), MATRIX_DATA_T, sender, this->_comm);
    return matrix::Dense(metadata, std::move(data));
}

void Communicator::gather_dense_into(matrix::Dense& mat,
                                     std::vector<int> displs,
                                     std::vector<int> recvcounts) {
    MPI_Allgatherv(MPI_IN_PLACE, recvcounts[this->rank()], MATRIX_DATA_T, mat.data_mut().data(),
                   recvcounts.data(), displs.data(), MATRIX_DATA_T, this->_comm);
}

std::optional<matrix::RawDenseParts> Communicator::gather_dense_parts(
    const matrix::Dense& mat,
    int target,
    const std::vector<int>& displs,
    const std::vector<int>& recvcounts) {
    if (this->is_coordinator()) {
        std::vector<matrix::Dense::data_t> data(mat.global_num_elems());
        MPI_Gatherv(mat.data().data(), mat.data().size(), MATRIX_DATA_T, data.data(),
                    recvcounts.data(), displs.data(), MATRIX_DATA_T, target, this->_comm);

        return matrix::RawDenseParts{
            mat.global_num_rows(),
            static_cast<field_t>(std::count_if(recvcounts.begin(), recvcounts.end(),
                                               [](const double x) { return x != 0; })),
            std::move(data)};

    } else {
        if (recvcounts[this->rank()] == 0) {
            MPI_Gatherv(nullptr, 0, MATRIX_DATA_T, nullptr, recvcounts.data(), displs.data(),
                        MATRIX_DATA_T, target, this->_comm);
        } else {
            MPI_Gatherv(mat.data().data(), mat.data().size(), MATRIX_DATA_T, nullptr,
                        recvcounts.data(), displs.data(), MATRIX_DATA_T, target, this->_comm);
        }
    }

    return std::nullopt;
}

void Communicator::send_sparse(const matrix::Sparse& mat, int target) {
    auto metadata = mat.get_metadata();
    MPI_Send(metadata.data(), metadata.size(), MATRIX_FIELD_T, target, 0, this->_comm);
    MPI_Send(mat.rows_offsets().data(), mat.rows_offsets().size(), MATRIX_FIELD_T, target, 0,
             this->_comm);

    MPI_Send(mat.cols_indices().data(), mat.cols_indices().size(), MATRIX_FIELD_T, target, 0,
             this->_comm);
    MPI_Send(mat.data().data(), mat.data().size(), MATRIX_DATA_T, target, 0, this->_comm);
}

matrix::Sparse Communicator::recv_sparse(int source) {
    metadata_t metadata(matrix::Sparse::num_metadata_fields);

    MPI_Recv(metadata.data(), metadata.size(), MATRIX_FIELD_T, source, 0, this->_comm,
             MPI_STATUS_IGNORE);

    std::vector<field_t> rows_offsets(metadata[matrix::Sparse::rows_offsets_size_metadata_idx]);
    std::vector<field_t> cols_indices(metadata[matrix::Sparse::cols_indices_size_metadata_idx]);
    std::vector<data_t> data(metadata[matrix::Sparse::data_size_metadata_idx]);

    MPI_Recv(rows_offsets.data(), rows_offsets.size(), MATRIX_FIELD_T, source, 0, this->_comm,
             MPI_STATUS_IGNORE);
    MPI_Recv(cols_indices.data(), cols_indices.size(), MATRIX_FIELD_T, source, 0, this->_comm,
             MPI_STATUS_IGNORE);
    MPI_Recv(data.data(), data.size(), MATRIX_DATA_T, source, 0, this->_comm, MPI_STATUS_IGNORE);

    return matrix::Sparse(metadata, std::move(rows_offsets), std::move(cols_indices),
                          std::move(data));
}

void Communicator::bcast_send_sparse(const matrix::Sparse& mat) {
    auto metadata = mat.get_metadata();
    MPI_Bcast(metadata.data(), metadata.size(), MATRIX_FIELD_T, this->_rank, this->_comm);
    MPI_Bcast((void*)mat.rows_offsets().data(), mat.rows_offsets().size(), MATRIX_FIELD_T,
              this->_rank, this->_comm);
    MPI_Bcast((void*)(mat.cols_indices().data()), mat.cols_indices().size(), MATRIX_FIELD_T,
              this->_rank, this->_comm);
    MPI_Bcast((void*)mat.data().data(), mat.data().size(), MATRIX_DATA_T, this->_rank, this->_comm);
}

matrix::Sparse Communicator::bcast_recv_sparse(int source) {
    metadata_t metadata(matrix::Sparse::num_metadata_fields);
    MPI_Bcast(metadata.data(), metadata.size(), MATRIX_FIELD_T, source, this->_comm);

    std::vector<field_t> rows_offsets(metadata[matrix::Sparse::rows_offsets_size_metadata_idx]);
    std::vector<field_t> cols_indices(metadata[matrix::Sparse::cols_indices_size_metadata_idx]);
    std::vector<data_t> data(metadata[matrix::Sparse::data_size_metadata_idx]);

    MPI_Bcast(rows_offsets.data(), rows_offsets.size(), MATRIX_FIELD_T, source, this->_comm);
    MPI_Bcast(cols_indices.data(), cols_indices.size(), MATRIX_FIELD_T, source, this->_comm);
    MPI_Bcast(data.data(), data.size(), MATRIX_DATA_T, source, this->_comm);

    return matrix::Sparse(metadata, std::move(rows_offsets), std::move(cols_indices),
                          std::move(data));
}

std::vector<std::vector<Communicator::field_t>> Communicator::gather_sparse_metadata(
    const matrix::Sparse& mat) {
    auto num_fields = matrix::Sparse::num_metadata_fields;
    std::vector<Communicator::field_t> metadata_flat(num_fields * this->world_size());

    MPI_Allgather(mat.get_metadata().data(), num_fields, MATRIX_FIELD_T, metadata_flat.data(),
                  num_fields, MATRIX_FIELD_T, this->_comm);
    std::vector<std::vector<field_t>> metadata;
    for (const auto& i : utils::range(this->world_size())) {
        metadata.push_back(std::vector<field_t>(metadata_flat.data() + i * num_fields,
                                                metadata_flat.data() + (i + 1) * num_fields));
    }
    return metadata;
}

void Communicator::isend_sparse(const matrix::Sparse& mat, int target) {
    this->_requests.emplace_back(MPI_Request{});
    MPI_Isend(mat.rows_offsets().data(), mat.rows_offsets().size(), MATRIX_FIELD_T, target, 0,
              this->_comm, &this->_requests[this->_requests.size() - 1]);

    this->_requests.emplace_back(MPI_Request{});
    MPI_Isend(mat.cols_indices().data(), mat.cols_indices().size(), MATRIX_FIELD_T, target, 0,
              this->_comm, &this->_requests[this->_requests.size() - 1]);

    this->_requests.emplace_back(MPI_Request{});
    MPI_Isend(mat.data().data(), mat.data().size(), MATRIX_DATA_T, target, 0, this->_comm,
              &this->_requests[this->_requests.size() - 1]);
}

void Communicator::irecv_sparse(matrix::Sparse& mat, int sender) {
    this->_requests.emplace_back(MPI_Request{});
    MPI_Irecv((void*)mat.rows_offsets().data(), mat.rows_offsets().size(), MATRIX_FIELD_T, sender,
              0, this->_comm, &this->_requests[this->_requests.size() - 1]);

    this->_requests.emplace_back(MPI_Request{});
    MPI_Irecv((void*)mat.cols_indices().data(), mat.cols_indices().size(), MATRIX_FIELD_T, sender,
              0, this->_comm, &this->_requests[this->_requests.size() - 1]);

    this->_requests.emplace_back(MPI_Request{});
    MPI_Irecv((void*)mat.data().data(), mat.data().size(), MATRIX_DATA_T, sender, 0, this->_comm,
              &this->_requests[this->_requests.size() - 1]);
}

void Communicator::wait_all() {
    auto statuses = std::make_unique<MPI_Status[]>(this->_requests.size());
    MPI_Waitall(this->_requests.size(), this->_requests.data(), statuses.get());
    this->_requests.clear();
}

int32_t Communicator::world_size() {
    return this->_world_size;
}

int32_t Communicator::rank() {
    return this->_rank;
}

bool Communicator::is_coordinator() {
    return this->_rank == 0;
}

std::optional<int> Communicator::reduce_int(int num, int root) {
    int result;
    MPI_Reduce(&num, &result, 1, MPI_INT32_T, MPI_SUM, root, this->_comm);
    if (this->rank() == root) {
        return result;
    } else {
        return std::nullopt;
    }
}
