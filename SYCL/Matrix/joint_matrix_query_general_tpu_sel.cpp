//==----- joint_matrix_query_general_tpu_sel.cpp  - DPC++ joint_matrix------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %clangxx -fsycl %s -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
public:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T>
void matrix_vnni(unsigned int rows, unsigned int cols, T *src, T *dest,
                 unsigned int vnniFactor) {
  for (unsigned int i = 0; i < rows / vnniFactor; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      for (unsigned int k = 0; k < vnniFactor; k++) {
        dest[i * cols * vnniFactor + j * vnniFactor + k] =
            src[(i * vnniFactor + k) * cols + j];
      }
    }
  }
}
static inline void native_cpuid(unsigned int *eax, unsigned int *ebx,
                                unsigned int *ecx, unsigned int *edx) {
  asm volatile("cpuid"
               : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
               : "0"(*eax), "2"(*ecx));
}

template <typename T1, typename T2, size_t M, size_t N, size_t K, size_t TM,
          size_t TN, size_t TK>
void matrix_multiply(queue q, big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                     big_matrix<T2, K / 4, N * 4> &B) {
  constexpr size_t SG_SZ = TN;
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<int8_t, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<int8_t, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<int32_t, 2> bufC(C.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]

         {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();

           joint_matrix<sub_group, T2, use::a, TM, TK, layout::row_major> sub_a;
           joint_matrix<sub_group, T2, use::b, TK, TN,
                        sycl::ext::intel::experimental::matrix::layout::packed>
               sub_b;
           joint_matrix<sub_group, T1, use::accumulator, TM, TN> sub_c;

           joint_matrix_load(sg, sub_c,
                             accC.get_pointer() + (sg_startx * TM) * N +
                                 sg_starty / SG_SZ * TN,
                             N, layout::row_major);
           for (int k = 0; k < K / TK; k += 1) {
             joint_matrix_load(
                 sg, sub_a, accA.get_pointer() + (sg_startx * TM) * K + k * TK,
                 K);
             joint_matrix_load(sg, sub_b,
                               accB.get_pointer() + (k * TK / 4) * (N * 4) +
                                   sg_starty / SG_SZ * TN * 4,
                               N * 4);
             sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(sg, sub_c,
                              accC.get_pointer() + (sg_startx * TM) * N +
                                  sg_starty / SG_SZ * TN,
                              N, layout::row_major);
         }); // parallel for
   }).wait();
}

bool is_this_amx_device(queue q) {
  auto dev_type =
      q.get_device().template get_info<sycl::info::device::device_type>();
  if (dev_type == info::device_type::cpu) {
    std::cout << "This is a CPU Device" << std::endl;
    unsigned eax, ebx, ecx, edx;

    eax = 1; /* processor info and feature bits */
    native_cpuid(&eax, &ebx, &ecx, &edx);
    int model = (eax >> 4) & 0xF;
    int family = (eax >> 8) & 0xF;
    int extended_model = (eax >> 16) & 0xF;
    int extended_family = (eax >> 20) & 0xFF;

    // SPR processsor means AMX is present
    if (model == 15 && family == 6 && extended_model == 8 &&
        extended_family == 0) {
      std::cout << "This is a SPR CPU Device" << std::endl;
      return true;
    }
  }
  return false;
}

bool is_this_xmx16_device(queue q) {
  auto dev_type =
      q.get_device().template get_info<sycl::info::device::device_type>();
  if (dev_type == info::device_type::gpu) {
    std::cout << "This is a GPU Device" << std::endl;
    auto dev_name =
        q.get_device().template get_info<sycl::info::device::name>();
    size_t found = dev_name.find("Intel(R) Graphics [0x0bd5]"); // PVC
    if (found != std::string::npos) {
      std::cout << "This is a PVC Device which contains XMX-16 TPU"
                << std::endl;
      return true;
    }
  }
  return false;
}

bool is_this_xmx8_device(queue q) {
  auto dev_type =
      q.get_device().template get_info<sycl::info::device::device_type>();
  if (dev_type == info::device_type::gpu) {
    std::cout << "This is a GPU Device" << std::endl;
    auto dev_name =
        q.get_device().template get_info<sycl::info::device::name>();
    size_t found = dev_name.find("Intel(R) Graphics [0x5610]"); // ATS-M
    if (found != std::string::npos) {
      std::cout << "This is a ATS-M Device which contains XMX-8 TPU"
                << std::endl;
      return true;
    }
  }
  return false;
}

template <tpu mytpu>
bool find_size_combinations(tpu_params<mytpu> params, int M, matrix_type Ta,
                            matrix_type Tb, matrix_type Tc) {
  for (int i = 0; params.num_combinations; i++) {
    if (Ta == params.combinations[i].atype &&
        Tb == params.combinations[i].btype &&
        Tc == params.combinations[i].accumulatortype) {
      if (params.combinations[i].msize == 0) {
        // this is max size query
        if (M <= params.combinations[i].max_msize)
          return true;
      } else {
        if (params.combinations[i].msize == M) {
          return true;
        }
      }
    }
  }
  return false;
}

template <tpu mytpu> bool is_sg_scope_supported(tpu_params<mytpu> params) {
  for (int i = 0; i < params.num_scopes; i++)
    if (params.scopes[i] == scope_t::sub_group)
      return true;
  return false;
}

template <typename T1, typename T2, size_t M, size_t N, size_t K>
int query_and_matrix_multiply(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                              big_matrix<T2, K / 4, N * 4> &B) {
  queue q;

  if (is_this_amx_device(q)) {
    using myparams = tpu_params<tpu::amx, int8_t, int8_t, int>;
    constexpr int TM = 4;
    constexpr int TN = myparams::N;
    constexpr int TK = myparams::K;
    tpu_params params = tpu_params<tpu::amx>();
    if (find_size_combinations(params, TM, matrix_type::sint8,
                               matrix_type::sint8, matrix_type::sint32) &&
        is_sg_scope_supported(params)) {
      std::cout << "M = 4 and sub-group scope is supported on AMX" << std::endl;
      matrix_multiply<T1, T2, M, N, K, TM, TN, TK>(q, C, A, B);
      return 1;
    } else
      return 2;
  }

  if (is_this_xmx16_device(q)) {
    using myparams = tpu_params<tpu::xmx16, int8_t, int8_t, int>;
    constexpr int TM = 4;
    constexpr int TN = myparams::N;
    constexpr int TK = myparams::K;
    tpu_params params = tpu_params<tpu::xmx16>();
    if (find_size_combinations(params, TM, matrix_type::sint8,
                               matrix_type::sint8, matrix_type::sint32) &&
        is_sg_scope_supported(params)) {
      std::cout << "M = 4 and sub-group scope is supported on XMX16 of PVC"
                << std::endl;
      matrix_multiply<T1, T2, M, N, K, TM, TN, TK>(q, C, A, B);
      return 1;
    } else
      return 2;
  }

  if (is_this_xmx8_device(q)) {
    using myparams = tpu_params<tpu::xmx8, int8_t, int8_t, int>;
    constexpr int TM = 4;
    constexpr int TN = myparams::N;
    constexpr int TK = myparams::K;
    tpu_params params = tpu_params<tpu::xmx16>();
    if (find_size_combinations(params, TM, matrix_type::sint8,
                               matrix_type::sint8, matrix_type::sint32) &&
        is_sg_scope_supported(params)) {
      std::cout << "M = 4 and sub-group scope is supported on XMX8 of ATS-M"
                << std::endl;
      matrix_multiply<T1, T2, M, N, K, TM, TN, TK>(q, C, A, B);
      return 1;
    } else
      return 2;
  }
  return 0;
}

void matrix_multiply_ref(int8_t *A, int8_t *B, int32_t *C, int M, int N,
                         int K) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
}

int main() {
  static constexpr size_t MATRIX_M = 128;
  static constexpr size_t MATRIX_N = 128;
  static constexpr size_t MATRIX_K = 128;
  static constexpr int vnniFactor = 4;
  int8_t A[MATRIX_M][MATRIX_K];
  int8_t B[MATRIX_K][MATRIX_N];
  int8_t Bvnni[MATRIX_K / 4][MATRIX_N * 4];
  int32_t C[MATRIX_M][MATRIX_N];
  int32_t D[MATRIX_M][MATRIX_N];

  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      A[i][j] = i + 2 * j;
    }
  }
  for (int i = 0; i < MATRIX_K; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      B[i][j] = i + j;
    }
  }
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      C[i][j] = 1;
      D[i][j] = 1;
    }
  }

  big_matrix<int32_t, MATRIX_M, MATRIX_N> MC((int32_t *)&C);
  big_matrix<int32_t, MATRIX_M, MATRIX_N> MD((int32_t *)&D);
  big_matrix<int8_t, MATRIX_M, MATRIX_K> MA((int8_t *)&A);
  matrix_vnni<int8_t>(MATRIX_K, MATRIX_N, (int8_t *)&B, (int8_t *)&Bvnni, 4);
  big_matrix<int8_t, MATRIX_K / 4, MATRIX_N * 4> MBvnni((int8_t *)&Bvnni);

  bool sel = query_and_matrix_multiply(MC, MA, MBvnni);
  bool res = true;
  if (sel == 1) {
    matrix_multiply_ref((int8_t *)&A, (int8_t *)&B, (int32_t *)&D, MATRIX_M,
                        MATRIX_N, MATRIX_K);

    for (int i = 0; i < MATRIX_M; i++) {
      for (int j = 0; j < MATRIX_N; j++) {
        if (C[i][j] != D[i][j])
          res = false;
      }
    }
    std::cout << (res ? "passed" : "failed") << std::endl;
  } else
    ((sel == 0) ? std::cout << "default device selector has no TPU" << std::endl
                : std::cout << "Selected combination is not supported by the "
                               "default device selector"
                            << std::endl);
  return !res;
}
