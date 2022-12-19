#define TM 8
#define TN SG_SZ
#define TK 16

static constexpr size_t M = TM * 2;
static constexpr size_t N = TN * 2;
static constexpr size_t K = TK * 2;

#define BF16_EPSILON 0.00781250

void matrix_multiply(queue q, float *C, bfloat16 *A, bfloat16 *B) {

  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;

  q.submit([&](handler &cgh) {
     cgh.parallel_for<class imatrix>(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]

         {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, bfloat16, use::a, TM, TK, layout::row_major>
               sub_a;
           // For B, we assume B has been already VNNIed.
           joint_matrix<sub_group, bfloat16, use::b, TK, TN,
                        ext::intel::experimental::matrix::layout::packed>
               sub_b;
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;

           joint_matrix_load(sg, sub_c,
                             C + (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
                             N, layout::row_major);
           for (int k = 0; k < K / TK; k += 1) { //
             joint_matrix_load(sg, sub_a, A + (sg_startx * TM) * K + k * TK, K);
             joint_matrix_load(sg, sub_b,
                               B + (k * TK / 2) * (N * 2) +
                                   sg_starty / SG_SZ * TN * 2,
                               N * 2);
             sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(sg, sub_c,
                              C + (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
                              N, layout::row_major);
         }); // parallel for
   }).wait();
}

float make_fp32(short x) {
  unsigned int y = x;
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

unsigned short make_bf16(float x) {
  int *res = reinterpret_cast<int *>(&x);
  *res = *res >> 16;
  return (unsigned short)*res;
}

void matrix_multiply_ref(int *A_mem, int *B_mem, int *C_mem, int M, int N,
                         int K) {
  // tiling
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        short *va = (short *)(A_mem + m * K + k);
        short *vb = (short *)(B_mem + k * N + n);
        float acc = *((float *)(C_mem + m * N + n));
        // FIXME: Should we do reduce-add in another version?
        for (int i = 0; i < 2; i++) {
          acc += (make_fp32(va[i]) * make_fp32(vb[i]));
        }
        *((float *)(C_mem + m * N + n)) = acc;
      }
    }
}

int main() {

  unsigned short Aref[M][K];
  unsigned short Bref[K / 2][N * 2];
  float Cref[M][N];
  float D[M][N];

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      Aref[i][j] = make_bf16(1.0f * (i + j));
    }
  }
  for (int i = 0; i < K / 2; i++) {
    for (int j = 0; j < N * 2; j++) {
      Bref[i][j] = make_bf16(2.0f * i + 3.0f * j);
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      D[i][j] = 1.0;
      Cref[i][j] = 1.0;
    }
  }

  queue q;

  bfloat16 *A = malloc_device<bfloat16>(M * K, q);
  bfloat16 *B = malloc_device<bfloat16>(K * N, q);
  float *C = malloc_device<float>(M * N, q);

  q.memcpy(A, Aref, sizeof(bfloat16) * M * K);
  q.memcpy(B, Bref, sizeof(bfloat16) * K * N);
  q.memcpy(C, D, sizeof(float) * M * N);

  matrix_multiply(q, C, A, B);
  q.memcpy(D, C, sizeof(float) * M * N);
  free(A, q);
  free(B, q);
  free(C, q);
  matrix_multiply_ref((int32_t *)Aref, (int32_t *)Bref, (int32_t *)Cref, M, N,
                      K / 2);

  bool res = true;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if ((fabs(D[i][j]) - fabs(Cref[i][j])) > BF16_EPSILON)
        res = false;
    }
  }
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
