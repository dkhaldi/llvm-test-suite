//==-------- element_wise_irreg_sum_rows.cpp  - DPC++ joint_matrix----- ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 8

#define TM 8
#define TN SG_SZ
#define TK 32

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
public:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T, size_t M, size_t N>
void sum_rows_ref(
    accessor<T, 2, access::mode::read, access::target::host_buffer> B,
    accessor<int, 1, access::mode::read, access::target::host_buffer>
        sum_rows) {
  int sum_rows_ref[8];
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      sum_rows_ref[i] += B[i][j];
    }
    auto diff = sum_rows[i] - sum_rows_ref[i];
    assert(std::fabs(static_cast<int>(diff)) <=
           std::numeric_limits<int>::epsilon());
  }
}

template <typename T, size_t M, size_t N>
void matrix_sum_rows(queue q, big_matrix<T, M, N> &B, nd_range<2> &r) {
  buffer<int8_t, 2> bufB(B.get_data(), range<2>(M, N));
  // size of vector is known because SG size of set by the user in this case
  buffer<int> sum_rows_v(8);

  q.submit([&](handler &cgh) {
     auto accB = bufB.get_access<access::mode::read_write>(cgh);
     auto v = sum_rows_v.get_access<access::mode::write>(cgh);
     cgh.parallel_for<class add_matrix>(
         r,
         [accB, v](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           ext::oneapi::sub_group sg = spmd_item.get_sub_group();
           joint_matrix<T, TK, TN, matrix_layout::packed_b> sub_b(sg);

           joint_matrix_load(sg, sub_b,
                             accB.get_pointer() + sg_starty / SG_SZ * TN * 4,
                             N * 4, matrix_layout::packed_b);
           // calculate sum of rows in sum_rows_v[8], there are 8 rows in sub_b
           // (tK/4)
           int32_t sum_local_rows[TK / 4]; //
           // sub_b has 32x8 elements, 32 elements per WI, 4 per WI per row
           auto data = sub_b.get_wi_data();
           // each WI calculates local sum of rows
           for (int row = 0; row < TK / 4; row++) { // there are 8 rows
             for (int i = 0; i < data.length() / SG_SZ; i++) { // 4 per row
               // i*SG_SIZE index is found based on the round robing
               // distribution we are using in the implementation
               // TODO: communicate this mapping information to the user using a
               // query interface
               sum_local_rows[row] += data[i * SG_SZ];
             }
             v[row] = reduce_over_group(sg, sum_local_rows[row], plus<>());
           }
         }); // parallel for
   }).wait();
  sum_rows_ref<T, M, N>(bufB.get_access<access::mode::read>(),
                        sum_rows_v.get_access<access::mode::read>());
}

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
int8_t B[MATRIX_M][MATRIX_N];

int main() {

  big_matrix<int8_t, MATRIX_M, MATRIX_N> MB((int8_t *)&B);

  size_t NDRangeM = MATRIX_M / TM;
  size_t NDRangeN = MATRIX_N / TN;
  queue q;
  nd_range<2> r({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ});

  for (int i = 0; i < MATRIX_M / 4; i++) {
    for (int j = 0; j < MATRIX_N * 4; j++) {
      B[i][j] = i;
    }
  }
  matrix_sum_rows<int8_t, MATRIX_M, MATRIX_N>(q, MB, r);

  return 0;
}
