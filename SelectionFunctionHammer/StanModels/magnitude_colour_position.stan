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
    matrix[M,M] inv_cholesky_m; // Cholesky factor in magnitude space
    matrix[C,C] inv_cholesky_c_T;     // Cholesky factor in colour space
}
parameters {
    matrix[M,C] a[H];
}
model {

    vector[H] log_prior;
    matrix[M,C] log_likelihood;

    // Prior
    for (h in 1:H){
        log_prior[h] = sum(square(inv_cholesky_m * (a[h] - mu[h]) * inv_cholesky_c_T));
    }
    
    // Loop over magnitude and colour
    for (m in 1:M){
        for (c in 1:C){

            // Local variables
            matrix[R,L] F;
            vector[P] x; // Probability in logit-space
            
            // Compute F
            for (l in 1:L) {
                F[:,l] = lambda[:,lower[l]:upper[l]] * to_vector(a[lower[l]:upper[l],M,C]);
            }
            
            // Compute x
            for (p in 1:P){
                x[p] = dot_product(F[pixel_to_ring[p]],azimuth[:,p]);
            }

            // Store log-likelihood
            log_likelihood[m,c] = binomial_logit_lupmf( k[m,c] | n[m,c], x );
            
        }  
    }

    target += -0.5*sum(log_prior ./ sigma) + sum(log_likelihood);
    
}
