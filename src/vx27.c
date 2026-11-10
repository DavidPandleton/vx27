#include "vx27_config.h"
#include "vx27_run.h"
#include "threadpool.h"
#include "quant_avx2.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
static int vx27_physical_cores(void) {
    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) return -1;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(len);
    if (!buf) return -1;
    if (!GetLogicalProcessorInformation(buf, &len)) { free(buf); return -1; }
    DWORD n = len / sizeof(*buf), cores = 0;
    for (DWORD i = 0; i < n; i++)
        if (buf[i].Relationship == RelationProcessorCore) cores++;
    free(buf);
    return (int)cores;
}
#else
static int vx27_physical_cores(void) { return -1; }
#endif

static int vx27_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

static void job_nop(void *arg, int tid, int nthreads) {
    (void)arg; (void)tid; (void)nthreads;
}

static void print_kv_row(const char *label, int ctx, double bytes_per_elem) {
    double gb = (double)VX27_KV_ELEMS_PER_TOK * ctx * bytes_per_elem / 1e9;
    printf("  KV %-8s ctx=%6d  %.3f GB\n", label, ctx, gb);
}

static int cmd_info(void) {
    printf("VORTEX-27 info (locked target: Qwen3.8-27B Heretic text-only)\n");
    printf("arch: %d layers = %d linear(Gated DeltaNet) + %d full(Gated Attn, every %d)\n",
        VX27_LAYERS, VX27_N_LINEAR, VX27_N_FULL, VX27_FULL_INTERVAL);
    printf("hidden=%d ffn=%d vocab=%d MTP_layers=%d\n",
        VX27_HIDDEN, VX27_FFN, VX27_VOCAB, VX27_MTP_LAYERS);
    printf("linear: QK=%d V=%d hd=%d conv=%d\n",
        VX27_LIN_QK_HEADS, VX27_LIN_V_HEADS, VX27_LIN_HEAD_DIM, VX27_LIN_CONV_DIM);
    printf("full: Q=%d KV=%d hd=%d rope_dim=%d theta=%.0f\n",
        VX27_ATTN_Q_HEADS, VX27_ATTN_KV_HEADS, VX27_ATTN_HEAD_DIM,
        VX27_ROPE_DIM, (double)VX27_ROPE_THETA);
    printf("ctx native=%d default=%d stretch=%d vision_dropped=%d\n",
        VX27_CTX_NATIVE, VX27_CTX_DEFAULT, VX27_CTX_STRETCH, VX27_VISION_DROPPED);

    int cores = vx27_physical_cores();
    printf("cpu: physical_cores=%d avx2_compiled=%d\n", cores, vx27_has_avx2());

    // threadpool smoke: 8 workers like ql/vortex ring
    char err[256] = {0};
    threadpool *tp = tp_create(8, err, sizeof(err));
    if (!tp) { printf("threadpool FAIL: %s\n", err); return 1; }
    tp_run(tp, job_nop, NULL);
    tp_free(tp);
    printf("threadpool: 8 workers OK (affinity via GetLogicalProcessorInformation)\n");

    // quant kernel linkage smoke (from ql, bit-identical)
    block_q4_k b4; memset(&b4, 0, sizeof(b4));
    block_q8_k b8; memset(&b8, 0, sizeof(b8));
    b8.qs[0] = 1;
    float d = dot_q4_k_q8_k_avx2(&b4, &b8, 1);
    printf("quant: dot_q4_k_q8_k_avx2 linked OK (zero-block dot=%.1f)\n", d);

    printf("KV per token: %d elems = 64.0 KB BF16\n", VX27_KV_ELEMS_PER_TOK);
    print_kv_row("BF16", 8192, 2.0);
    print_kv_row("BF16", VX27_CTX_DEFAULT, 2.0);
    print_kv_row("BF16", VX27_CTX_STRETCH, 2.0);
    print_kv_row("Q4", VX27_CTX_DEFAULT, 0.5);
    print_kv_row("Q4", VX27_CTX_STRETCH, 0.5);

    printf("budget: core=%.1fGB dual + head=%.1fGB vram + kv<=%.1fGB single\n",
        VX27_CORE_GB, VX27_HEAD_GB, VX27_KV_BUDGET_GB);
    double kv64_q4 = (double)VX27_KV_ELEMS_PER_TOK * VX27_CTX_DEFAULT * 0.5 / 1e9;
    double kv128_q4 = (double)VX27_KV_ELEMS_PER_TOK * VX27_CTX_STRETCH * 0.5 / 1e9;
    printf("check: 64K Q4 KV=%.2fGB %s | 128K Q4 KV=%.2fGB %s\n",
        kv64_q4, kv64_q4 <= VX27_KV_BUDGET_GB ? "FIT" : "OVER",
        kv128_q4, kv128_q4 <= VX27_KV_BUDGET_GB ? "FIT" : "OVER");
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && (!strcmp(argv[1], "info") || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
        if (!strcmp(argv[1], "info")) return cmd_info();
        printf("usage: vx27.exe info | run \"prompt\" [opsi] | chat [opsi]\n");
        return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "run")) return vx27_run_main(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "chat")) return vx27_chat_main(argc, argv);
    printf("usage: vx27.exe info | run \"prompt\" [opsi] | chat [opsi]\n");
    return 1;
}
