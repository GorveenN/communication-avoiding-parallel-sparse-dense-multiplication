#include <argparse.h>
#include <matmul.h>
#include <range.h>
#include <utils.h>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    Args args = Args{argc, argv};
    std::optional<matrix::RawDenseParts> result;
    std::unique_ptr<matmul::MatmulAlg> alg = nullptr;

    if (args.inner_algorithm) {
        alg = std::make_unique<matmul::InnerABC>(args);
    } else {
        alg = std::make_unique<matmul::ColA>(args);
    }

    alg->run();
    result = alg->collect_denses();

    if (args.verbose && result) {
        std::cout << result.value();
    }

    if (args.ge_value) {
        auto ge = alg->result_num_greater_than(args.ge_value.value());
        if (ge) {
            std::cout << ge.value() << std::endl;
        }
    }

    return 0;
}
