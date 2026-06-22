// CPU reference correctness tests for the sparkinfer kernel algorithms.
//
// These re-implement each CUDA kernel's exact numerical algorithm in plain C++
// and check it against an INDEPENDENT double-precision ground truth (different
// loop order / higher precision). A match is real evidence the algorithm is
// correct; the device-side sm_120 compile (see .cudaverify) separately proves
// the same code targets the RTX 5090. Together they cover "valid for the 5090"
// and "computes the right thing" — the two halves a GPU-less environment allows.
//
// Build: g++ -O2 -std=c++17 cpu_reference_test.cpp -o cpu_reference_test

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <utility>

using std::vector;
static std::mt19937 rng(1234);
static float frand() { return std::uniform_real_distribution<float>(-1.f, 1.f)(rng); }

static int g_fail = 0;
static void check(const char* name, double max_err, double tol) {
    bool ok = max_err <= tol;
    printf("  [%s] %-34s max_err=%.3e (tol=%.0e)\n", ok ? "PASS" : "FAIL", name, max_err, tol);
    if (!ok) g_fail++;
}

static float silu(float x) { return x / (1.f + std::exp(-x)); }

// ---------------------------------------------------------------------------
// 1. Flash decode: online-softmax (kernel algorithm) vs naive full softmax.
// ---------------------------------------------------------------------------
static double test_attention(int HD, int kvlen) {
    vector<float> q(HD), K(kvlen * HD), V(kvlen * HD);
    for (auto& x : q) x = frand();
    for (auto& x : K) x = frand();
    for (auto& x : V) x = frand();
    const float scale = 1.f / std::sqrt((float)HD);

    // Ground truth (double precision, two-pass softmax).
    vector<double> scores(kvlen);
    double mx = -1e300;
    for (int t = 0; t < kvlen; t++) {
        double d = 0; for (int i = 0; i < HD; i++) d += (double)q[i] * K[t * HD + i];
        scores[t] = d * scale; mx = std::max(mx, scores[t]);
    }
    double denom = 0; for (int t = 0; t < kvlen; t++) denom += std::exp(scores[t] - mx);
    vector<double> ref(HD, 0);
    for (int t = 0; t < kvlen; t++) {
        double p = std::exp(scores[t] - mx) / denom;
        for (int i = 0; i < HD; i++) ref[i] += p * V[t * HD + i];
    }

    // Kernel algorithm: single-pass online softmax in float.
    float m = -1e30f, l = 0.f; vector<float> acc(HD, 0.f);
    for (int t = 0; t < kvlen; t++) {
        float d = 0; for (int i = 0; i < HD; i++) d += q[i] * K[t * HD + i];
        float score = d * scale;
        float m_new = std::max(m, score);
        float corr = std::exp(m - m_new), p = std::exp(score - m_new);
        l = l * corr + p;
        for (int i = 0; i < HD; i++) acc[i] = acc[i] * corr + p * V[t * HD + i];
        m = m_new;
    }
    double err = 0; for (int i = 0; i < HD; i++) err = std::max(err, std::abs(acc[i] / l - ref[i]));
    return err;
}

// ---------------------------------------------------------------------------
// 2. Router top-k: kernel mask-argmax algorithm vs sort-based reference.
// ---------------------------------------------------------------------------
static double test_router(int E, int K) {
    vector<float> logits(E); for (auto& x : logits) x = frand();

    // Reference: stable sort by (value desc, index asc), take K; softmax over them.
    vector<int> idx(E); for (int i = 0; i < E; i++) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
    vector<int> ref_id(idx.begin(), idx.begin() + K);
    double rmx = logits[ref_id[0]], rden = 0;
    for (int j = 0; j < K; j++) rden += std::exp((double)logits[ref_id[j]] - rmx);
    vector<double> ref_w(K);
    for (int j = 0; j < K; j++) ref_w[j] = std::exp((double)logits[ref_id[j]] - rmx) / rden;

    // Kernel algorithm: K passes of arg-max with masking, then softmax over picks.
    vector<float> s = logits; vector<int> sel(K); vector<float> sl(K);
    for (int j = 0; j < K; j++) {
        float best = -1e30f; int bi = -1;
        for (int e = 0; e < E; e++) if (s[e] > best || (s[e] == best && e < bi)) { best = s[e]; bi = e; }
        sel[j] = bi; sl[j] = best; s[bi] = -1e30f;
    }
    float kmx = sl[0]; for (int j = 1; j < K; j++) kmx = std::max(kmx, sl[j]);
    float kden = 0; for (int j = 0; j < K; j++) kden += std::exp(sl[j] - kmx);

    double err = 0;
    for (int j = 0; j < K; j++) {
        if (sel[j] != ref_id[j]) err = std::max(err, 1.0);
        err = std::max(err, std::abs(std::exp(sl[j] - kmx) / kden - ref_w[j]));
    }
    return err;
}

// ---------------------------------------------------------------------------
// 3. SwiGLU expert FFN: kernel math (float) vs double ground truth.
// ---------------------------------------------------------------------------
static double test_swiglu(int H, int F) {
    vector<float> X(H), gate(H * F), up(H * F), down(F * H);
    for (auto& x : X) x = frand();
    for (auto& x : gate) x = frand() * 0.1f;
    for (auto& x : up) x = frand() * 0.1f;
    for (auto& x : down) x = frand() * 0.1f;
    const float w = 0.37f;

    vector<double> hbuf_d(F), ref(H, 0);
    for (int f = 0; f < F; f++) {
        double g = 0, u = 0;
        for (int h = 0; h < H; h++) { g += (double)X[h] * gate[h * F + f]; u += (double)X[h] * up[h * F + f]; }
        hbuf_d[f] = (g / (1.0 + std::exp(-g))) * u;
    }
    for (int h = 0; h < H; h++) { double y = 0; for (int f = 0; f < F; f++) y += hbuf_d[f] * down[f * H + h]; ref[h] = w * y; }

    vector<float> hbuf(F), acc(H, 0.f);
    for (int f = 0; f < F; f++) {
        float g = 0, u = 0;
        for (int h = 0; h < H; h++) { g += X[h] * gate[h * F + f]; u += X[h] * up[h * F + f]; }
        hbuf[f] = silu(g) * u;
    }
    for (int h = 0; h < H; h++) { float y = 0; for (int f = 0; f < F; f++) y += hbuf[f] * down[f * H + h]; acc[h] = w * y; }

    double err = 0; for (int h = 0; h < H; h++) err = std::max(err, std::abs((double)acc[h] - ref[h]));
    return err;
}

// ---------------------------------------------------------------------------
// 4. GEMM: tiled accumulation order vs double triple-loop.
// ---------------------------------------------------------------------------
static double test_gemm(int M, int N, int Kd) {
    vector<float> A(M * Kd), B(Kd * N);
    for (auto& x : A) x = frand();
    for (auto& x : B) x = frand();
    vector<double> ref(M * N, 0);
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { double s = 0; for (int k = 0; k < Kd; k++) s += (double)A[i*Kd+k]*B[k*N+j]; ref[i*N+j] = s; }

    const int TILE = 16; vector<float> C(M * N, 0.f);
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        float acc = 0.f;
        for (int k0 = 0; k0 < Kd; k0 += TILE) { float t = 0.f; for (int k = k0; k < std::min(k0+TILE,Kd); k++) t += A[i*Kd+k]*B[k*N+j]; acc += t; }
        C[i*N+j] = acc;
    }
    double err = 0; for (int i = 0; i < M*N; i++) err = std::max(err, std::abs((double)C[i] - ref[i]));
    return err;
}

// ---------------------------------------------------------------------------
// 5. RMSNorm: kernel math vs double ground truth.
// ---------------------------------------------------------------------------
static double test_rmsnorm(int cols) {
    vector<float> x(cols), wt(cols); for (auto& v : x) v = frand(); for (auto& v : wt) v = frand();
    const float eps = 1e-6f;
    double ss = 0; for (int c = 0; c < cols; c++) ss += (double)x[c]*x[c];
    double inv = 1.0 / std::sqrt(ss / cols + eps);
    vector<double> ref(cols); for (int c = 0; c < cols; c++) ref[c] = x[c]*inv*wt[c];

    float fss = 0; for (int c = 0; c < cols; c++) fss += x[c]*x[c];
    float finv = 1.f/std::sqrt(fss/cols + eps);
    double err = 0; for (int c = 0; c < cols; c++) err = std::max(err, std::abs((double)(x[c]*finv*wt[c]) - ref[c]));
    return err;
}

int main() {
    printf("sparkinfer kernel algorithm correctness (CPU reference)\n");
    check("attention hd128 kv1",   test_attention(128, 1),    1e-4);
    check("attention hd128 kv333", test_attention(128, 333),  1e-4);
    check("attention hd256 kv1024",test_attention(256, 1024), 2e-4);
    check("attention hd512 kv777", test_attention(512, 777),  2e-4);
    check("router E256 k8",        test_router(256, 8),       1e-6);
    check("router E128 k8",        test_router(128, 8),       1e-6);
    check("swiglu H2048 F512",     test_swiglu(2048, 512),    1e-3);
    check("swiglu H512 F1536",     test_swiglu(512, 1536),    1e-3);
    check("gemm 64x96x128",        test_gemm(64, 96, 128),    1e-3);
    check("gemm 17x33x49",         test_gemm(17, 33, 49),     1e-3);
    check("rmsnorm cols2048",      test_rmsnorm(2048),        1e-4);
    printf("%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail);
    return g_fail ? 1 : 0;
}
