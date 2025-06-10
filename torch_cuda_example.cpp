#include <cstdlib>
#include <iostream>

#include <dlfcn.h>

#include <torch/torch.h>

bool load_torch_cuda_library() {
    static void *handle =
        dlopen("/root/pytorch/dist/lib/libtorch_cuda.so", RTLD_NOW);
    return handle != nullptr;
}

int main() {
    // Check if CUDA is available
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available! Please make sure you have a "
                     "CUDA-capable GPU and PyTorch with CUDA support."
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Load the PyTorch CUDA library
    if (!load_torch_cuda_library()) {
        std::cerr << "Failed to load the PyTorch CUDA library. Please check "
                     "the library path."
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "CUDA is available! Using GPU for tensor operations."
              << std::endl;
    std::cout << "Number of CUDA devices: " << torch::cuda::device_count()
              << std::endl;

    // Create device object for CUDA
    torch::Device device(torch::kCUDA);

    // Create two tensors on CUDA device
    torch::Tensor tensor1 = torch::rand({3, 4}, device);
    torch::Tensor tensor2 = torch::rand({3, 4}, device);

    std::cout << "\nTensor 1 (on CUDA):" << std::endl;
    std::cout << tensor1 << std::endl;

    std::cout << "\nTensor 2 (on CUDA):" << std::endl;
    std::cout << tensor2 << std::endl;

    // Add the tensors
    torch::Tensor result = tensor1 + tensor2;

    std::cout << "\nResult of tensor1 + tensor2:" << std::endl;
    std::cout << result << std::endl;

    // Additional operations to demonstrate CUDA functionality
    torch::Tensor sum_result = torch::sum(result);
    std::cout << "\nSum of all elements in result tensor:" << std::endl;
    std::cout << sum_result << std::endl;

    return EXIT_SUCCESS;
}
