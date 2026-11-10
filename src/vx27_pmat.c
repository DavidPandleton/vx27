#include "vx27_pmat.h"
#include "vx27_gpu_reg.h"
#include "quant.h"
#include "vx27_bp3.h"
#include <immintrin.h>
#include <string.h>

int vx27_gpu_reg_matvec(const void *W, int rows, int cols, const float *x, float *y, double *ms);

// dot int8[n] x float[n], n kelipatan 8
static float dot_i8_f32(const int8_t *q, const float *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256i qi = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q + i)));
        __m256 qv = _mm256_cvtepi32_ps(qi);
        __m256 xv = _mm256_loadu_ps(x + i);
#if defined(__FMA__)
        acc = _mm256_fmadd_ps(qv, xv, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(qv, xv));
#endif
    }
    float t[8]; _mm256_storeu_ps(t, acc);
    return t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
}

// In-register BP3 unpack+dot, 8 bobot per iterasi (tanpa buffer int8).
// Plane p[b*4 + w] = 32 bobot bit-b; grup 8 dgn k%8==0 selalu dalam 1 word.
static float bp3_dot_reg(const vx27_bp3_128 *g, int ng, const float *x) {
    __m256 acc = _mm256_setzero_ps();
    const __m256i cnt8 = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i three = _mm256_set1_epi32(3);
    const __m256i eight = _mm256_set1_epi32(8);
    for (int k = 0; k < ng; k++) {
        uint16_t h = g[k].scale_h; uint32_t u = h;
        uint32_t sign=(u&0x8000u)<<16, exp=(u>>10)&0x1Fu, mant=u&0x3FFu, bits;
        if (exp==0) bits=mant? (sign|((exp+1+112u)<<23)|(mant<<13)):sign;
        else if (exp==31) bits=sign|0x7F800000u|(mant<<13);
        else bits=sign|((exp+112u)<<23)|(mant<<13);
        float sc; memcpy(&sc,&bits,4);
        __m256 scv = _mm256_set1_ps(sc);
        for (int i = 0; i < 128; i += 8) {
            int w = i >> 5; // u32 word utk bit i (0..3)
            uint32_t w0 = g[k].p[0*4+w], w1 = g[k].p[1*4+w], w2 = g[k].p[2*4+w];
            __m256i cnt = _mm256_add_epi32(cnt8, _mm256_set1_epi32(i & 31));
            __m256i b0 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)w0), cnt), one);
            __m256i b1 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)w1), cnt), one);
            __m256i b2 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)w2), cnt), one);
            __m256i uu = _mm256_or_si256(b0, _mm256_or_si256(_mm256_slli_epi32(b1, 1), _mm256_slli_epi32(b2, 2)));
            __m256i isneg = _mm256_cmpgt_epi32(uu, three);
            __m256i qi = _mm256_blendv_epi8(uu, _mm256_sub_epi32(uu, eight), isneg);
            __m256 qv = _mm256_mul_ps(_mm256_cvtepi32_ps(qi), scv);
            __m256 xv = _mm256_loadu_ps(x + k*128 + i);
#if defined(__FMA__)
            acc = _mm256_fmadd_ps(qv, xv, acc);
#else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(qv, xv));
#endif
        }
    }
    float t[8]; _mm256_storeu_ps(t, acc);
    return t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
}

typedef struct { const uint8_t *W; uint32_t dtype; const float *x; float *y; int rows, cols; } pmj;

// Jalur skalar (referensi verifikasi test): unpack ke int8 tmp lalu dot AVX2.
static float bp3_dot_scalar(const vx27_bp3_128 *g, int ng, const float *x) {
    double s = 0;
    int8_t tmp[128];
    for (int k = 0; k < ng; k++) {
        uint16_t h = g[k].scale_h;
        uint32_t u = h, sign=(u&0x8000u)<<16, exp=(u>>10)&0x1Fu, mant=u&0x3FFu, bits;
        if (exp==0) bits=mant? (sign|((exp+1+112u)<<23)|(mant<<13)):sign;
        else if (exp==31) bits=sign|0x7F800000u|(mant<<13);
        else bits=sign|((exp+112u)<<23)|(mant<<13);
        float sc; memcpy(&sc,&bits,4);
        for (int i = 0; i < 128; i++) {
            uint32_t uu=((g[k].p[0*4+i/32]>>(i%32))&1)|(((g[k].p[1*4+i/32]>>(i%32))&1)<<1)|(((g[k].p[2*4+i/32]>>(i%32))&1)<<2);
            tmp[i] = (int8_t)((uu>=4)?(int)uu-8:(int)uu);
        }
        s += (double)dot_i8_f32(tmp, x + k*128, 128) * sc;
    }
    return (float)s;
}

float vx27_pmat_bp3_scalar(const void *row, const float *x, int cols); // utk test
float vx27_pmat_bp3_scalar(const void *row, const float *x, int cols) {
    return bp3_dot_scalar((const vx27_bp3_128*)row, cols / 128, x);
}
float vx27_pmat_bp3_reg(const void *row, const float *x, int cols); // utk test
float vx27_pmat_bp3_reg(const void *row, const float *x, int cols) {
    return bp3_dot_reg((const vx27_bp3_128*)row, cols / 128, x);
}

static float row_dot(const uint8_t *row, uint32_t dtype, const float *x, int cols) {
    if (dtype == 0) {
        const float *r = (const float*)row;
        double s = 0; for (int c = 0; c < cols; c++) s += (double)r[c]*x[c];
        return (float)s;
    } else if (dtype == 2) {
        const block_q8_k *b = (const block_q8_k*)row;
        int nb = cols / 256;
        double s = 0;
        for (int k = 0; k < nb; k++) s += (double)dot_i8_f32(b[k].qs, x + k*256, 256) * b[k].d;
        return (float)s;
    } else { // 4 = BP3_128 -> jalur in-register produksi
        return bp3_dot_reg((const vx27_bp3_128*)row, cols / 128, x);
    }
}

static size_t row_bytes(uint32_t dtype, int cols) {
    if (dtype == 0) return (size_t)cols * 4;
    if (dtype == 2) return (size_t)(cols / 256) * sizeof(block_q8_k);
    return (size_t)(cols / 128) * sizeof(vx27_bp3_128);
}

static void pm_worker(void *a, int tid, int nth) {
    pmj *j = (pmj*)a;
    size_t rb = row_bytes(j->dtype, j->cols);
    int rs = (int)((long long)j->rows * tid / nth), re = (int)((long long)j->rows * (tid+1) / nth);
    for (int r = rs; r < re; r++)
        j->y[r] = row_dot(j->W + (size_t)r * rb, j->dtype, j->x, j->cols);
}

// ---- batched matvec: satu pass bobot utk B vektor (teacher-forcing) ----
// Urutan operasi per vektor == jalur single (bp3_dot_reg / row_dot dgn
// double-acc utk Q8 / FMA AVX2 utk F32) sehingga hasil bit-identik.
#define VX27_PMAT_BMAX 8

// BP3 per baris: unpack 8-wide sekali, fmadd ke B akumulasi.
static void bp3_row_batch(const vx27_bp3_128 *g, int ng, const float *xb,
                          int cols, float *yrow, int B) {
    __m256 acc[VX27_PMAT_BMAX];
    for (int b = 0; b < B; b++) acc[b] = _mm256_setzero_ps();
    const __m256i cnt8 = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i three = _mm256_set1_epi32(3);
    const __m256i eight = _mm256_set1_epi32(8);
    for (int k = 0; k < ng; k++) {
        uint16_t h = g[k].scale_h; uint32_t u = h;
        uint32_t sign=(u&0x8000u)<<16, exp=(u>>10)&0x1Fu, mant=u&0x3FFu, bits;
        if (exp==0) bits=mant? (sign|((exp+1+112u)<<23)|(mant<<13)):sign;
        else if (exp==31) bits=sign|0x7F800000u|(mant<<13);
        else bits=sign|((exp+112u)<<23)|(mant<<13);
        float sc; memcpy(&sc,&bits,4);
        __m256 scv = _mm256_set1_ps(sc);
        for (int i = 0; i < 128; i += 8) {
            int w = i >> 5;
            uint32_t w0 = g[k].p[0*4+w], w1 = g[k].p[1*4+w], w2 = g[k].p[2*4+w];
            __m256i cnt = _mm256_add_epi32(cnt8, _mm256_set1_epi32(i & 31));
            __m256i b0 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)w0), cnt), one);
            __m256i b1 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)w1), cnt), one);
            __m256i b2 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)w2), cnt), one);
            __m256i uu = _mm256_or_si256(b0, _mm256_or_si256(_mm256_slli_epi32(b1, 1), _mm256_slli_epi32(b2, 2)));
            __m256i isneg = _mm256_cmpgt_epi32(uu, three);
            __m256i qi = _mm256_blendv_epi8(uu, _mm256_sub_epi32(uu, eight), isneg);
            __m256 qv = _mm256_mul_ps(_mm256_cvtepi32_ps(qi), scv);
            const float *xp0 = xb + (size_t)k*128 + i; // batch slot 0
            for (int b = 0; b < B; b++) {
                __m256 xv = _mm256_loadu_ps(xp0 + (size_t)b*cols);
#if defined(__FMA__)
                acc[b] = _mm256_fmadd_ps(qv, xv, acc[b]);
#else
                acc[b] = _mm256_add_ps(acc[b], _mm256_mul_ps(qv, xv));
#endif
            }
        }
    }
    for (int b = 0; b < B; b++) {
        float t[8]; _mm256_storeu_ps(t, acc[b]);
        yrow[b] = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    }
}

typedef struct { const uint8_t *W; uint32_t dtype; const float *xb; float *yb; int rows, cols, B; } pmBj;

static void pm_batch_worker(void *a, int tid, int nth) {
    pmBj *j = (pmBj*)a;
    size_t rb = row_bytes(j->dtype, j->cols);
    int rs = (int)((long long)j->rows * tid / nth), re = (int)((long long)j->rows * (tid+1) / nth);
    int B = j->B, cols = j->cols;
    for (int r = rs; r < re; r++) {
        float yrow[VX27_PMAT_BMAX];
        if (j->dtype == 4) {
            bp3_row_batch((const vx27_bp3_128*)(j->W + (size_t)r * rb), cols / 128, j->xb, cols, yrow, B);
        } else if (j->dtype == 2) {
            const block_q8_k *b8 = (const block_q8_k*)(j->W + (size_t)r * rb);
            int nb = cols / 256;
            double s[VX27_PMAT_BMAX] = {0,0,0,0,0,0,0,0};
            for (int k = 0; k < nb; k++) {
                for (int b = 0; b < B; b++)
                    s[b] += (double)dot_i8_f32(b8[k].qs, j->xb + (size_t)b*cols + k*256, 256) * b8[k].d;
            }
            for (int b = 0; b < B; b++) yrow[b] = (float)s[b];
        } else {
            // F32: akumulasi double serial per vektor == row_dot single
            // (bit-identik; jalur F32 bukan hot path produksi).
            const float *wr = (const float*)(j->W + (size_t)r * rb);
            double s[VX27_PMAT_BMAX] = {0,0,0,0,0,0,0,0};
            for (int c = 0; c < cols; c++) {
                double w = (double)wr[c];
                for (int b = 0; b < B; b++) s[b] += w * (double)j->xb[(size_t)b*cols + c];
            }
            for (int b = 0; b < B; b++) yrow[b] = (float)s[b];
        }
        for (int b = 0; b < B; b++) j->yb[(size_t)b*j->rows + r] = yrow[b];
    }
}

void vx27_pmat_batch(threadpool *tp, const void *W, uint32_t dtype,
               const float *xb, float *yb, int rows, int cols, int B) {
    if (B < 1) return;
    if (B == 1) { // degenerasi: identik single
        vx27_pmat(tp, W, dtype, xb, yb, rows, cols);
        return;
    }
    if (B > VX27_PMAT_BMAX) B = VX27_PMAT_BMAX;
    if (tp && rows >= 256) {
        pmBj j = {(const uint8_t*)W, dtype, xb, yb, rows, cols, B};
        tp_run(tp, pm_batch_worker, &j);
    } else {
        pmBj j = {(const uint8_t*)W, dtype, xb, yb, rows, cols, B};
        pm_batch_worker(&j, 0, 1);
    }
}

void vx27_pmat(threadpool *tp, const void *W, uint32_t dtype,
               const float *x, float *y, int rows, int cols) {
    // Jalur GPU device-resident (hanya BP3; Q8/F32 tetap CPU).
    // Gagal/tidak terdaftar -> jatuh ke CPU di bawah (tidak pernah fatal).
    if (dtype == 4) {
        double m = 0;
        if (vx27_gpu_reg_matvec(W, rows, cols, x, y, &m) == 1) return;
    }
    if (tp && rows >= 256) {
        pmj j = {(const uint8_t*)W, dtype, x, y, rows, cols};
        tp_run(tp, pm_worker, &j);
    } else {
        pmj j = {(const uint8_t*)W, dtype, x, y, rows, cols};
        pm_worker(&j, 0, 1);
    }
}
