data {
    int<lower=0> J;        // number of pixels
    int<lower=0> N;        // number of HEALPix isolatitude rings
    int<lower=0> H;        // number of harmonics
    int<lower=0> L;        // 2 * max l of hamonics + 1
    matrix[N,H] lambda;    // spherical harmonics decomposed
    matrix[L,J] azimuth;   // spherical harmonics decomposed
    int indices[J];        // map N->J
    int lower[L];          // map H->L
    int upper[L];          // map H->L
    int k[J];
    int n[J];
}
parameters {
    vector[H] a;
}
transformed parameters {
    vector[J] x;
    {
        matrix[N,L] F;
        for (i in 1:L) {
            F[:,i] = lambda[:,lower[i]:upper[i]] * a[lower[i]:upper[i]];
        }
        for (i in 1:J) {
            x[i] = F[indices[i]] * azimuth[:,i];
        }
    }
}
model {
    a ~ std_normal();
    k ~ binomial_logit(n, x);
}
