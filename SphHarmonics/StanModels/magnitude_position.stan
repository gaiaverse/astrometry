data {
    int<lower=0> J;        // number of pixels
    int<lower=0> N;        // number of HEALPix isolatitude rings
    int<lower=0> H;        // number of harmonics
    int<lower=0> L;        // 2 * max l of hamonics + 1
    int<lower=0> Nm;      // number of bins in magnitude space
    matrix[N,H] lambda;    // spherical harmonics decomposed
    matrix[L,J] azimuth;   // spherical harmonics decomposed
    matrix[Nm,Nm] Lm;     // Cholesky factor in magnitude space
    int indices[J];        // map N->J
    int lower[L];          // map H->L
    int upper[L];          // map H->L
    int k[Nm,J];
    int n[Nm,J];
    real sigmal[H];
}
parameters {
    vector[H] z[Nm];
}
transformed parameters {
    row_vector[J] x[Nm];
    {
        vector[H] a[Nm];
        matrix[N,L] F;
        matrix[L,J] G;
        
        // Form a
        for (i in 1:Nm){
            for (j in 1:H){
                a[i,j] = sigmal[j] * Lm[i] * to_vector(z[:,j]);
            }
        }
        
        // Compute x
        for (m in 1:Nm){
            for (i in 1:L) {
                F[:,i] = lambda[:,lower[i]:upper[i]] * a[m,lower[i]:upper[i]];
            }
            G = transpose(F[indices]);
            x[m] = columns_dot_product(G,azimuth);
        }
    }
}
model {
    for (m in 1:Nm){
        z[m] ~ std_normal();
        k[m] ~ binomial_logit(n[m], x[m]);
    }
}
