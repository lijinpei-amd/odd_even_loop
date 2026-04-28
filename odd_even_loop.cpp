#include <torch/extension.h>

#include <ATen/ceil_div.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <c10/util/TypeCast.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace {

constexpr const char* kKernelName =
    "Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname1_gfx950";

constexpr uint32_t kBlockM = 256;
constexpr uint32_t kBlockN = 256;
constexpr uint32_t kBlockThreads = 256;

struct __attribute__((packed, aligned(8))) KernelArgs {
  uint32_t gemm_info;
  uint32_t kernel_info0;
  uint32_t kernel_info1;
  uint32_t num_wg;
  uint32_t size_i;
  uint32_t size_j;
  uint32_t size_k;
  uint32_t size_l;
  void* D;
  void* C;
  void* A;
  void* B;
  uint32_t stride_d1;
  uint32_t stride_dk;
  uint32_t stride_c1;
  uint32_t stride_ck;
  uint32_t stride_a0;
  uint32_t stride_ak;
  uint32_t stride_b1;
  uint32_t stride_bk;
  float alpha;
  float beta;
};

static_assert(sizeof(KernelArgs) == 104, "unexpected Tensile kernarg size");

#define HIP_CHECK_THROW(expr)                                                 \
  do {                                                                        \
    hipError_t _err = (expr);                                                 \
    TORCH_CHECK(_err == hipSuccess, "HIP error from ", #expr, ": ",          \
                hipGetErrorString(_err));                                     \
  } while (0)

void check_bf16_cuda_2d(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.defined(), name, " is undefined");
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA/HIP tensor");
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, name,
              " must have dtype torch.bfloat16");
  TORCH_CHECK(t.dim() == 2, name, " must be 2D");
}

void validate_ab(const at::Tensor& a, const at::Tensor& b) {
  check_bf16_cuda_2d(a, "a");
  check_bf16_cuda_2d(b, "b");
  TORCH_CHECK(a.get_device() == b.get_device(),
              "a and b must be on the same device");

  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b.size(1);
  TORCH_CHECK(b.size(0) == K, "b shape must be (K, N); got b.size(0)=",
              b.size(0), " while a.size(1)=", K);
  TORCH_CHECK(M % kBlockM == 0, "M must be a multiple of ", kBlockM,
              " for this no-edge assembly kernel, got ", M);
  TORCH_CHECK(N % kBlockN == 0, "N must be a multiple of ", kBlockN,
              " for this no-edge assembly kernel, got ", N);
  TORCH_CHECK(K % 64 == 0,
              "K must be a multiple of 64 for this MT256x256x64 kernel, got ",
              K);
  TORCH_CHECK(a.stride(1) == 1,
              "a must have contiguous K dimension; expected a.stride(1)=1, got ",
              a.stride(1));
  TORCH_CHECK(b.stride(0) == 1,
              "b must be the transposed-B view with contiguous K dimension; "
              "expected b.stride(0)=1, got ",
              b.stride(0));
}

void validate_d_against_ab(const at::Tensor& a, const at::Tensor& b,
                           const at::Tensor& d) {
  check_bf16_cuda_2d(d, "d");
  TORCH_CHECK(d.get_device() == a.get_device(),
              "d must be on the same device as a and b");

  const int64_t M = a.size(0);
  const int64_t N = b.size(1);
  TORCH_CHECK(d.size(0) == M && d.size(1) == N,
              "d shape must be (M, N); expected (", M, ", ", N, "), got (",
              d.size(0), ", ", d.size(1), ")");
  TORCH_CHECK(d.stride(0) == 1,
              "d must be column-major/Fortran-strided; expected d.stride(0)=1, got ",
              d.stride(0));
  TORCH_CHECK(d.stride(1) >= M,
              "d.stride(1) must be at least M; expected >= ", M, ", got ",
              d.stride(1));
}

class LoadedKernel {
 public:
  LoadedKernel(std::string hsaco_path, std::string kernel_name)
      : hsaco_path_(std::move(hsaco_path)), kernel_name_(std::move(kernel_name)) {
    HIP_CHECK_THROW(hipModuleLoad(&module_, hsaco_path_.c_str()));
    HIP_CHECK_THROW(
        hipModuleGetFunction(&function_, module_, kernel_name_.c_str()));
  }

  LoadedKernel(const LoadedKernel&) = delete;
  LoadedKernel& operator=(const LoadedKernel&) = delete;

  ~LoadedKernel() {
    if (module_ != nullptr) {
      (void)hipModuleUnload(module_);
    }
  }

  hipFunction_t function() const { return function_; }
  const std::string& hsaco_path() const { return hsaco_path_; }
  const std::string& kernel_name() const { return kernel_name_; }

 private:
  std::string hsaco_path_;
  std::string kernel_name_;
  hipModule_t module_ = nullptr;
  hipFunction_t function_ = nullptr;
};

std::mutex g_kernel_mutex;
std::unique_ptr<LoadedKernel> g_kernel;
std::atomic<LoadedKernel*> g_kernel_fast{nullptr};

LoadedKernel& get_kernel(const std::string& hsaco_path,
                         const std::string& kernel_name) {
  LoadedKernel* fast = g_kernel_fast.load(std::memory_order_acquire);
  if (fast != nullptr && fast->hsaco_path() == hsaco_path &&
      fast->kernel_name() == kernel_name) {
    return *fast;
  }

  std::lock_guard<std::mutex> guard(g_kernel_mutex);
  if (!g_kernel || g_kernel->hsaco_path() != hsaco_path ||
      g_kernel->kernel_name() != kernel_name) {
    g_kernel = std::make_unique<LoadedKernel>(hsaco_path, kernel_name);
    g_kernel_fast.store(g_kernel.get(), std::memory_order_release);
  }
  return *g_kernel;
}

void launch_impl(const at::Tensor& a, const at::Tensor& b, const at::Tensor& d,
                 const std::string& hsaco_path,
                 const std::string& kernel_name) {
  c10::cuda::OptionalCUDAGuard device_guard(a.device());

  const uint32_t M = c10::checked_convert<uint32_t>(a.size(0), "M");
  const uint32_t K = c10::checked_convert<uint32_t>(a.size(1), "K");
  const uint32_t N = c10::checked_convert<uint32_t>(b.size(1), "N");
  const uint32_t batch = 1;
  const uint32_t num_wg =
      at::ceil_div<uint32_t>(M, kBlockM) * at::ceil_div<uint32_t>(N, kBlockN);

  const uint32_t d_stride1 =
      c10::checked_convert<uint32_t>(d.stride(1), "d.stride(1)");
  const uint32_t d_numel = c10::checked_convert<uint32_t>(d.numel(), "d.numel()");

  KernelArgs args{};
  args.gemm_info = 1;
  args.kernel_info0 = 1;
  args.kernel_info1 = 16;
  args.num_wg = num_wg;
  args.size_i = M;
  args.size_j = N;
  args.size_k = batch;
  args.size_l = K;
  args.D = d.data_ptr();
  args.C = d.data_ptr();
  args.A = const_cast<void*>(a.data_ptr());
  args.B = const_cast<void*>(b.data_ptr());
  args.stride_d1 = d_stride1;
  args.stride_dk = d_numel;
  args.stride_c1 = d_stride1;
  args.stride_ck = d_numel;
  args.stride_a0 = c10::checked_convert<uint32_t>(a.stride(0), "a.stride(0)");
  args.stride_ak = c10::checked_convert<uint32_t>(a.numel(), "a.numel()");
  args.stride_b1 = c10::checked_convert<uint32_t>(b.stride(1), "b.stride(1)");
  args.stride_bk = c10::checked_convert<uint32_t>(b.numel(), "b.numel()");
  args.alpha = 1.0f;
  args.beta = 0.0f;

  LoadedKernel& loaded = get_kernel(hsaco_path, kernel_name);
  size_t arg_size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                    &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &arg_size,
                    HIP_LAUNCH_PARAM_END};

  hipStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  HIP_CHECK_THROW(hipModuleLaunchKernel(loaded.function(), num_wg, 1, 1,
                                        kBlockThreads, 1, 1, 0, stream,
                                        nullptr, config));
}

}  // namespace

at::Tensor empty_output(const at::Tensor& a, const at::Tensor& b) {
  validate_ab(a, b);
  return at::empty_strided({a.size(0), b.size(1)}, {1, a.size(0)},
                           a.options());
}

at::Tensor matmul(const at::Tensor& a, const at::Tensor& b,
                  const std::string& hsaco_path,
                  const std::string& kernel_name = kKernelName) {
  at::Tensor d = empty_output(a, b);
  validate_d_against_ab(a, b, d);
  launch_impl(a, b, d, hsaco_path, kernel_name);
  return d;
}

at::Tensor matmul_out(const at::Tensor& a, const at::Tensor& b,
                      const at::Tensor& d,
                      const std::string& hsaco_path,
                      const std::string& kernel_name = kKernelName) {
  validate_ab(a, b);
  validate_d_against_ab(a, b, d);
  launch_impl(a, b, d, hsaco_path, kernel_name);
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("empty_output", &empty_output, py::arg("a"), py::arg("b"));
  m.def("matmul", &matmul, py::arg("a"), py::arg("b"), py::arg("hsaco_path"),
        py::arg("kernel_name") = kKernelName);
  m.def("matmul_out", &matmul_out, py::arg("a"), py::arg("b"), py::arg("d"),
        py::arg("hsaco_path"), py::arg("kernel_name") = kKernelName);
}
