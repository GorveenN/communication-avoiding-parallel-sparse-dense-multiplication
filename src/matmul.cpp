#include <communicator.h>
#include <matmul.h>
#include <parser.h>
#include <utils.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <thread>

#include <chrono>
#include <thread>

// MatmulAlg
namespace matmul {

MatmulAlg::MatmulAlg(const Args& args)
    : args(args),
      dense_(nullptr),
      result_(nullptr),
      sparse_(nullptr),
      spare_sparse_(std::make_unique<matrix::Sparse>()),
      global_comm_(args.argc, args.argv),
      repl_group_comm_(this->global_comm_, args.repl_group_size, false),
      repl_plane_comm_(this->global_comm_, args.repl_group_size, true) {}

void MatmulAlg::gather_sparse_metadata() {
    this->metadata_ = this->repl_plane_comm_.gather_sparse_metadata(*this->sparse_);
}

void MatmulAlg::prepare_data() {
    this->prepare_matrices();
    this->replicate_sparse();
    this->gather_sparse_metadata();
}

void MatmulAlg::prepare_matrices() {
    this->read_and_distribute_sparse();
    this->generate_dense();
    this->generate_result_matrix();
}

void MatmulAlg::read_and_distribute_sparse() {
    if (this->global_comm_.is_coordinator()) {
        if (!utils::file_exists(this->args.sparse_matrix_file)) {
            throw std::runtime_error(std::string("File ") + this->args.sparse_matrix_file +
                                     " does not exists");
        }
        std::ifstream sparse_stream{this->args.sparse_matrix_file};
        matrix::Sparse whole_sparse = Parser::parse(sparse_stream);

        auto splitted_sparse = whole_sparse.into_parts(this->global_comm_.world_size(),
                                                       this->sparse_partitioned_rowwise());
        this->sparse_ = std::make_unique<matrix::Sparse>(std::move(splitted_sparse[0]));
        for (int32_t i = 1; i < (int32_t)splitted_sparse.size(); ++i) {
            this->global_comm_.send_sparse(splitted_sparse[i], i);
        }
    } else {
        this->sparse_ = std::make_unique<matrix::Sparse>(this->global_comm_.recv_sparse(0));
    }
}
void MatmulAlg::replicate_sparse() {
    std::vector<matrix::Sparse> parts;
    for (auto sender = 0; sender < this->repl_group_comm_.world_size(); ++sender) {
        if (sender == this->repl_group_comm_.rank()) {
            this->repl_group_comm_.bcast_send_sparse(*this->sparse_);
            parts.emplace_back(std::move(*this->sparse_));
        } else {
            parts.emplace_back(this->repl_group_comm_.bcast_recv_sparse(sender));
        }
    }
    this->sparse_ =
        std::make_unique<matrix::Sparse>(matrix::Sparse(parts, this->sparse_partitioned_rowwise()));
}

void MatmulAlg::generate_dense() {
    int32_t size = this->sparse_->global_num_cols();
    auto [cols_beg_idx, length] = utils::partition_number_nth(size, this->dense_plane_num_chunks(),
                                                              this->dense_plane_chunk_idx());

    this->dense_ = std::make_unique<matrix::Dense>(
        size, size, 0, size, cols_beg_idx, cols_beg_idx + length, this->args.seed_for_dense_matrix);
}

void MatmulAlg::generate_result_matrix() {
    this->result_ = std::make_unique<matrix::Dense>(matrix::Dense::like(*this->dense_));
}

void MatmulAlg::schedule_shift_sparse(int step) {
    auto [source, target, recv_from_chunk_idx] = this->sparse_shift_indices(step);
    this->spare_sparse_->set_metadata(this->metadata_[recv_from_chunk_idx]);
    std::unique_ptr<matrix::Sparse> new_sparse = nullptr;
    this->repl_plane_comm_.isend_sparse(*this->sparse_, target);
    this->repl_plane_comm_.irecv_sparse(*this->spare_sparse_, source);
}

void MatmulAlg::finish_shift_sparse() {
    this->repl_plane_comm_.wait_all();
    this->spare_sparse_.swap(this->sparse_);
}

bool MatmulAlg::is_coordinator() {
    return this->global_comm_.is_coordinator();
}

std::optional<matrix::RawDenseParts> MatmulAlg::collect_denses() {
    auto [displs, recvcounts] =
        utils::partition_number(this->dense_->global_num_cols(), this->dense_num_chunks());

    displs.resize(this->global_comm_.world_size(), displs[displs.size() - 1]);
    recvcounts.resize(this->global_comm_.world_size(), 0);

    for (auto& x : displs) {
        x *= this->dense_->global_num_rows();
    }

    for (auto& x : recvcounts) {
        x *= this->dense_->global_num_rows();
    }

    return this->global_comm_.gather_dense_parts(*this->result_, 0, displs, recvcounts);
}

std::optional<int> MatmulAlg::result_num_greater_than(double ge) {
    int to_send = 0;
    if (this->should_gather_dense()) {
        to_send = std::count_if(this->result_->data().begin(), this->result_->data().end(),
                                [=](const auto& x) { return x >= ge; });
    }

    return this->global_comm_.reduce_int(to_send, 0);
}

}  // namespace matmul

// InnerABC
namespace matmul {

int InnerABC::dense_num_chunks() {
    return this->repl_plane_comm_.world_size();
}

bool InnerABC::should_gather_dense() {
    return this->global_comm_.rank() < this->repl_plane_comm_.world_size();
}

InnerABC::InnerABC(const Args& args)
    : MatmulAlg(args),
      repl_plane_group_comm_(
          this->global_comm_,
          this->global_comm_.rank() % (this->global_comm_.world_size() / args.repl_group_size)) {}

std::pair<std::vector<int>, std::vector<int>> InnerABC::dense_shift_displ_recv() {
    auto [offsets, sizes] =
        utils::partition_number(this->dense_->global_num_cols(), this->global_comm_.world_size());

    auto sum_chunk_size = this->global_comm_.world_size() / this->args.repl_group_size;

    std::vector<int> reduced_offsets;
    std::vector<int> reduced_sizes;

    for (const auto& i : utils::range(offsets.size() / sum_chunk_size)) {
        reduced_offsets.emplace_back(offsets[i * sum_chunk_size] * this->dense_->local_num_cols());
        reduced_sizes.emplace_back(this->dense_->local_num_cols() *
                                   std::accumulate(sizes.begin() + i * sum_chunk_size,
                                                   sizes.begin() + (i + 1) * sum_chunk_size, 0,
                                                   std::plus<>()));
    }

    return std::make_pair(reduced_offsets, reduced_sizes);
}

void InnerABC::run() {
    this->prepare_data();

    for (const auto& _ : utils::range(this->args.exponent)) {
        for (const auto& i :
             utils::range(this->repl_plane_comm_.world_size() / this->args.repl_group_size)) {
            this->schedule_shift_sparse(i);
            matmul_sparse_dense(*this->sparse_, *this->dense_, *this->result_);
            this->finish_shift_sparse();
        }

        this->sync_dense();
        this->dense_.swap(this->result_);
        this->result_->zero_data();
    }
    this->dense_.swap(this->result_);
}

std::tuple<int, int, int> InnerABC::sparse_shift_indices(int step) {
    auto micro_group_world_size = this->repl_plane_comm_.world_size() / this->args.repl_group_size;
    auto micro_group_rank = this->repl_plane_comm_.rank() % micro_group_world_size;
    auto micro_group_offset =
        micro_group_world_size * (this->repl_plane_comm_.rank() / micro_group_world_size);
    auto micro_group_rank_minus_step = micro_group_rank - (step + 1);
    auto micro_recv_from_chunk_idx = micro_group_rank_minus_step >= 0
                                         ? micro_group_rank_minus_step
                                         : micro_group_world_size + micro_group_rank_minus_step;

    auto micro_source = micro_group_rank == 0 ? micro_group_world_size - 1
                                              : (micro_group_rank - 1) % micro_group_world_size;
    auto micro_target = (micro_group_rank + 1) % micro_group_world_size;

    return std::make_tuple(micro_source + micro_group_offset, micro_target + micro_group_offset,
                           micro_recv_from_chunk_idx + micro_group_offset);
}

void InnerABC::sync_dense() {
    auto [displs, recvcounts] = this->dense_shift_displ_recv();

    this->repl_plane_group_comm_.gather_dense_into(*this->result_, displs, recvcounts);
}

int InnerABC::dense_plane_num_chunks() {
    return this->repl_plane_comm_.world_size();
}

int InnerABC::dense_plane_chunk_idx() {
    return this->global_comm_.rank() % this->repl_plane_comm_.world_size();
}

bool InnerABC::sparse_partitioned_rowwise() {
    return true;
}

// ColA
ColA::ColA(const Args& args) : MatmulAlg(args) {}

int ColA::dense_plane_num_chunks() {
    return this->global_comm_.world_size();
}

int ColA::dense_plane_chunk_idx() {
    return this->global_comm_.rank();
}

bool ColA::sparse_partitioned_rowwise() {
    return false;
}

void ColA::run() {
    this->prepare_data();

    for (auto rep = 0; rep < this->args.exponent; rep++) {
        for (auto i = 0; i < this->repl_plane_comm_.world_size(); i++) {
            this->schedule_shift_sparse(i);
            matmul_sparse_dense(*this->sparse_, *this->dense_, *this->result_);
            this->finish_shift_sparse();
        }
        this->dense_.swap(this->result_);
        this->result_->zero_data();
    }
    // std::this_thread::sleep_for(std::chrono::seconds(this->global_comm_.rank()));
    // spdlog::info("{} {} {} \n{}\n{}", this->global_comm_.rank(), this->repl_group_comm_.rank(),
    //              this->repl_plane_comm_.rank(), *this->sparse_, *this->dense_);
    this->dense_.swap(this->result_);

    // std::this_thread::sleep_for(
    //     std::chrono::seconds(this->global_comm_.world_size() - this->global_comm_.rank()));
}

int ColA::dense_num_chunks() {
    return this->global_comm_.world_size();
}

bool ColA::should_gather_dense() {
    return true;
}

std::tuple<int, int, int> ColA::sparse_shift_indices(int step) {
    auto rank_minus_step = this->repl_plane_comm_.rank() - (step + 1);
    auto recv_from_chunk_idx = rank_minus_step >= 0
                                   ? rank_minus_step
                                   : this->repl_plane_comm_.world_size() + rank_minus_step;

    auto source = this->repl_plane_comm_.rank() == 0
                      ? this->repl_plane_comm_.world_size() - 1
                      : (this->repl_plane_comm_.rank() - 1) % this->repl_plane_comm_.world_size();
    auto target = (this->repl_plane_comm_.rank() + 1) % this->repl_plane_comm_.world_size();
    return std::make_tuple(source, target, recv_from_chunk_idx);
}
};  // namespace matmul
