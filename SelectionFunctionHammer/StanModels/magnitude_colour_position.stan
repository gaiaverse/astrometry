data {
    int<lower=0> P;                       // number of pixels
    int<lower=0> M;                       // number of bins in magnitude space
    int<lower=0> M_subspace;              // number of inducing points in magnitude space
    int<lower=0> C;                       // number of bins in colour space
    int<lower=0> C_subspace;              // number of inducing points in colour space
    int<lower=0> L;                       // 2 * max l of hamonics + 1
    int<lower=0> H;                       // number of harmonics
    int<lower=0> R;                       // number of HEALPix isolatitude rings
    matrix[R,H] lambda;                   // spherical harmonics decomposed
    matrix[L,P] azimuth;                  // spherical harmonics decomposed
    int pixel_to_ring[P];                 // map P->R
    int lower[L];                         // map H->L
    int upper[L];                         // map H->L
    vector[H] mu;                         // mean of each harmonic
    vector[H] sigma;                      // sigma of each harmonic
    int k[M,C,P];                         // number of heads
    int n[M,C,P];                         // number of flips
    row_vector[M_subspace] cholesky_m[M]; // Cholesky factor in magnitude space
    vector[C_subspace] cholesky_c[C];     // Cholesky factor in colour space
}
parameters {
    matrix[M_subspace,C_subspace] z[H];
}
transformed parameters {

    vector[P] x[M,C]; // Probability in logit-space
    
    { // Local environment to keep a and F out of the output
        vector[H] a;
        matrix[R,L] F;
    
        // Loop over magnitude and colour
        for (m in 1:M){
            for (c in 1:C){
                
                // Compute a
                for (h in 1:H){
                    a[h] = mu[h] + sigma[h] * cholesky_m[m] * z[h] * cholesky_c[c];
                }
                
                // Compute F
                for (l in 1:L) {
                    F[:,l] = lambda[:,lower[l]:upper[l]] * a[lower[l]:upper[l]];
                }
                
                // Compute x
                for (p in 1:P){
                    x[m,c,p] = dot_product(F[pixel_to_ring[p]],azimuth[:,p]);
                }
                
            }  
        }
    }
}
model {
    
    real log_likelihood[M,C];

    // Prior
    for (h in 1:H){
        to_vector(z[h]) ~ std_normal();
    }
    
    // Likelihood
    for (m in 1:M){
        for (c in 1:C){
            log_likelihood[m,c] = binomial_logit_pmf( k[m,c] | n[m,c], x[m,c] );
        }
    }

    target += sum(log_likelihood);
    
}
