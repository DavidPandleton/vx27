#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "../src/vx27_deltanet.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

// RNG deterministik (LCG 64-bit) agar test reproduksibel tanpa platform-rand.
static uint64_t rng_s = 0x123456789ABCDEFULL;
static double rnd(void) { rng_s = rng_s * 6364136223846793005ULL + 1442695040888963407ULL; return (double)(rng_s >> 11) * (1.0 / 9007199254740992.0); }

// Oracle independen (double penuh), transkripsi langsung rumus referensi
// torch_recurrent_gated_delta_rule: l2norm(q,k) [eps di DALAM sqrt, tanpa /n],
// q/=sqrt(hd); decay=exp(g); kv=S^T k; d=(v-kv)*beta; S=S*decay+k⊗d; o=S^T q.
static void oracle_step(double *S, const double *q, const double *k, const double *v,
                        double beta, double g_log, double *o, int hd) {
    double qn[256], kn[256], qs = 1.0 / sqrt((double)hd), dec = exp(g_log), kv[256];
    double s = 0.0; for (int i = 0; i < hd; i++) s += q[i]*q[i];
    s = 1.0 / sqrt(s + 1e-6); for (int i = 0; i < hd; i++) qn[i] = q[i]*s*qs;
    s = 0.0; for (int i = 0; i < hd; i++) s += k[i]*k[i];
    s = 1.0 / sqrt(s + 1e-6); for (int i = 0; i < hd; i++) kn[i] = k[i]*s;
    for (int i = 0; i < hd; i++) for (int j = 0; j < hd; j++) S[i*hd+j] *= dec;
    for (int j = 0; j < hd; j++) { double a = 0.0; for (int i = 0; i < hd; i++) a += S[i*hd+j]*kn[i]; kv[j] = a; }
    for (int i = 0; i < hd; i++) for (int j = 0; j < hd; j++)
        S[i*hd+j] += kn[i]*((v[j]-kv[j])*beta);
    for (int j = 0; j < hd; j++) { double a = 0.0; for (int i = 0; i < hd; i++) a += S[i*hd+j]*qn[i]; o[j] = a; }
}

static double oracle_beta(double b) { return 1.0 / (1.0 + exp(-b)); }
static double oracle_g(double a, double Alog, double dtb) {
    double x = a + dtb, sp = x > 20.0 ? x : (x < -20.0 ? exp(x) : log(1.0 + exp(x)));
    return -exp(Alog) * sp;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const int hd = 128;
    // 1) gate mapping: nilai eksplisit per referensi (beta=sigmoid(b) TANPA dt_bias;
    //    g=-exp(Alog)*softplus(a+dt_bias), termasuk rezim ekstrem softplus).
    {
        struct { double a, Al, b, dt; } cs[] = {
            {0.1, 0.5, -0.3, 1.0}, {3.0, -1.2, 2.0, 1.0}, {-25.0, 0.0, 0.0, 1.0},
            {25.0, 2.0, -2.0, 1.0}, {0.0, -5.0, 5.0, -3.0}, {-30.0, 1.0, 30.0, 30.0},
        };
        double md = 0;
        for (size_t c = 0; c < sizeof(cs)/sizeof(cs[0]); c++) {
            double be = 0, gg = 0, bo = oracle_beta(cs[c].b), go = oracle_g(cs[c].a, cs[c].Al, cs[c].dt);
            vx27_deltanet_gate_params(cs[c].a, cs[c].Al, cs[c].b, cs[c].dt, &be, &gg);
            double d = fabs(be-bo); if (d > md) md = d;
            d = fabs(gg-go); if (d > md) md = d;
            if (!(be > 0.0 && be < 1.0 && gg <= 0.0)) { printf("FAIL: rentang gate case %zu\n", c); fails++; }
        }
        printf("  gate mapping maxdiff=%.3e\n", md);
        CHECK(md == 0.0, "beta/g == referensi bit-identik");
    }

    // 2) exact vs oracle: 40 trial acak, T=1 dan rantai T=6.
    {
        double md1 = 0, mdS = 0; int fin = 1;
        for (int trial = 0; trial < 40; trial++) {
            float Sf[16384], qf[128], kf[128], vf[128], of[128];
            double Sd[16384], qd[128], kd[128], vd[128], od[128];
            for (int i = 0; i < hd*hd; i++) { double v = (rnd()*2.0-1.0)*0.5; Sf[i] = (float)v; Sd[i] = v; }
            for (int i = 0; i < hd; i++) {
                qd[i] = qf[i] = (float)(rnd()*2.0-1.0);
                kd[i] = kf[i] = (float)(rnd()*2.0-1.0);
                vd[i] = vf[i] = (float)(rnd()*2.0-1.0)*2.0;
            }
            double beta = 0.05 + 0.9*rnd();
            double g = -8.0*rnd(); // decay in (e^-8, 1]
            // T=1
            oracle_step(Sd, qd, kd, vd, beta, g, od, hd);
            vx27_deltanet_step_exact(Sf, qf, kf, vf, beta, g, of, hd);
            for (int i = 0; i < hd; i++) { double d = fabs((double)of[i]-od[i]); if (d > md1) md1 = d; }
            for (int i = 0; i < hd*hd; i++) { double d = fabs((double)Sf[i]-Sd[i]); if (d > mdS) mdS = d; }
            for (int i = 0; i < hd; i++) if (!(of[i] > -1e30 && of[i] < 1e30)) fin = 0;
            // rantai T=6 dengan input baru tiap langkah
            for (int t = 1; t < 6; t++) {
                for (int i = 0; i < hd; i++) {
                    qd[i] = qf[i] = (float)(rnd()*2.0-1.0);
                    kd[i] = kf[i] = (float)(rnd()*2.0-1.0);
                    vd[i] = vf[i] = (float)(rnd()*2.0-1.0)*2.0;
                }
                beta = 0.05 + 0.9*rnd(); g = -8.0*rnd();
                oracle_step(Sd, qd, kd, vd, beta, g, od, hd);
                vx27_deltanet_step_exact(Sf, qf, kf, vf, beta, g, of, hd);
                for (int i = 0; i < hd; i++) { double d = fabs((double)of[i]-od[i]); if (d > md1) md1 = d; }
                for (int i = 0; i < hd*hd; i++) { double d = fabs((double)Sf[i]-Sd[i]); if (d > mdS) mdS = d; }
            }
        }
        printf("  exact-vs-oracle 40x(1+5 langkah): maxdiff out=%.3e state=%.3e\n", md1, mdS);
        CHECK(fin, "output finite semua trial");
        CHECK(md1 < 1e-4 && mdS < 1e-3, "exact == oracle (toleransi float)");
    }

    // determinisme bit-identik
    {
        float S1[16384] = {0}, S2[16384] = {0}, q[128], k[128], v[128], o1[128], o2[128];
        for (int i = 0; i < hd; i++) { q[i] = (float)i*0.01f-0.5f; k[i] = (float)(i%7)*0.03f; v[i] = (float)(i%5)*0.07f; }
        vx27_deltanet_step_exact(S1, q, k, v, 0.7, -0.5, o1, hd);
        vx27_deltanet_step_exact(S2, q, k, v, 0.7, -0.5, o2, hd);
        double md = 0; for (int i = 0; i < hd; i++) { double d = fabs((double)o1[i]-o2[i]); if (d > md) md = d; }
        CHECK(md == 0.0, "step deterministik bit-identik");
    }

    // 3) conv carryover vs konvolusi naif penuh (zero left-pad), T=8.
    {
        const int C = 32, T = 8, K = 4;
        float *w = malloc((size_t)C*K*4);
        float *xs = malloc((size_t)T*C*4);
        for (int i = 0; i < C*K; i++) w[i] = (float)(rnd()*2.0-1.0)*0.3f;
        for (int i = 0; i < T*C; i++) xs[i] = (float)(rnd()*2.0-1.0);
        float *st = calloc((size_t)C*3, 4), *y = malloc((size_t)C*4), *yn = malloc((size_t)C*4);
        double md = 0; int fin = 1;
        for (int t = 0; t < T; t++) {
            vx27_conv1d_step(st, xs + (size_t)t*C, w, y, C, 1);
            // naif: window [t-3..t] zero-padded kiri, tap j=0 terlama
            for (int c = 0; c < C; c++) {
                double acc = 0.0;
                for (int j = 0; j < K; j++) {
                    int tt = t - (K-1) + j;
                    double xv = tt < 0 ? 0.0 : xs[(size_t)tt*C + c];
                    acc += (double)w[(size_t)c*K + j] * xv;
                }
                double sv = acc / (1.0 + exp(-acc)); // silu
                yn[c] = (float)sv;
                double d = fabs((double)y[c]-yn[c]); if (d > md) md = d;
                if (!(y[c] > -1e30 && y[c] < 1e30)) fin = 0;
            }
        }
        printf("  conv carryover vs naif T=%d C=%d: maxdiff=%.3e\n", T, C, md);
        CHECK(fin, "conv finite");
        CHECK(md < 1e-5f, "conv carryover == konvolusi naif");
        free(w); free(xs); free(st); free(y); free(yn);
    }

    // 4) kuantifikasi selisih vs kernel lama (INFORMATIF, bukan assert nilai):
    //    kernel lama: tanpa l2norm, tanpa skala 1/sqrt, beta+dt_bias, g tanpa softplus.
    {
        const int hd2 = 128;
        float Sold[16384], Snew[16384], q[128], k[128], v[128], oo[128], on[128];
        for (int i = 0; i < hd2*hd2; i++) { Sold[i] = Snew[i] = (float)(rnd()*2.0-1.0)*0.3f; }
        for (int i = 0; i < hd2; i++) { q[i] = (float)(rnd()*2.0-1.0); k[i] = (float)(rnd()*2.0-1.0); v[i] = (float)(rnd()*2.0-1.0); }
        double b = 0.4, a = 0.2, Al = 0.7, dt = 1.0, be, gg;
        vx27_deltanet_gate_params(a, Al, b, dt, &be, &gg);
        // lama: bh=sigmoid(b+dt), ah=exp(-exp(Al+a)) — bandingkan retrieve mentah
        vx27_deltanet_update_ref(Sold, k, v, (float)exp(-exp(Al+a)), hd2);
        // skala manual ala lama? kernel lama tidak punya skala: bandingkan retrieve mentah
        vx27_deltanet_retrieve_ref(Sold, q, oo, hd2);
        vx27_deltanet_step_exact(Snew, q, k, v, be, gg, on, hd2);
        double md = 0; for (int i = 0; i < hd2; i++) { double d = fabs((double)oo[i]-on[i]); if (d > md) md = d; }
        printf("  kernel lama vs exact (sintetik): maxdiff=%.3e (diharapkan BESAR: l2norm/skala/gate beda)\n", md);
        CHECK(oo[0] == oo[0] && on[0] == on[0], "keduanya finite");
    }

    if (fails) { printf("test_deltanet_exact: %d FAIL\n", fails); return 1; }
    printf("test_deltanet_exact: PASS\n");
    return 0;
}
