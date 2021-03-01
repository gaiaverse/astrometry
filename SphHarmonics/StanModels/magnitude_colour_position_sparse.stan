data {
    int<lower=0> P;           // number of pixels
    int<lower=0> M;           // number of bins in magnitude space
    int<lower=0> C;           // number of bins in colour space
    int<lower=0> L;           // 2 * max l of hamonics + 1
    int<lower=0> H;           // number of harmonics
    int<lower=0> R;           // number of HEALPix isolatitude rings
    matrix[R,H] lambda;       // spherical harmonics decomposed
    matrix[L,P] azimuth;      // spherical harmonics decomposed
    int pixel_to_ring[P];     // map P->R
    int lower[L];             // map H->L
    int upper[L];             // map H->L
    matrix[M,M] cholesky_m;   // Cholesky factor in magnitude space
    matrix[C,C] cholesky_c; // Cholesky factor in colour space transposed
    int cholesky_nonzero_m;   // number of non-zero elements in cholesky_m
    int cholesky_nonzero_c;   // number of non-zero elements in cholesky_m
    vector[H] mu;             // mean of each harmonic
    vector[H] sigma;          // sigma of each harmonic
    int k[M,C,P];             // number of heads
    int n[M,C,P];             // number of flips
}
transformed data {
    
    row_vector[cholesky_nonzero_m] cholesky_w_m = to_row_vector(csr_extract_w(cholesky_m));
    int cholesky_v_m[cholesky_nonzero_m] = csr_extract_v(cholesky_m);
    int cholesky_u_m[M+1] = csr_extract_u(cholesky_m);
    
    vector[cholesky_nonzero_c] cholesky_w_c = csr_extract_w(cholesky_c);
    int cholesky_v_c[cholesky_nonzero_c] = csr_extract_v(cholesky_c);
    int cholesky_u_c[C+1] = csr_extract_u(cholesky_c);
    
}
parameters {
    matrix[M,C] z[H];
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
                    //a[h] = mu[h] + sigma[h] * cholesky_m[m,:] * z[h] * cholesky_c_T[:,c];
                    a[h] = mu[h] + sigma[h] * (cholesky_w_m[cholesky_u_m[m]:cholesky_u_m[m+1]-1] * z[h,cholesky_v_m[cholesky_u_m[m]:cholesky_u_m[m+1]-1], cholesky_v_c[cholesky_u_c[c]:cholesky_u_c[c+1]-1]] * cholesky_w_c[cholesky_u_c[c]:cholesky_u_c[c+1]-1]);
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

    // Prior
    for (h in 1:H){
        to_vector(z[h]) ~ std_normal();
    }
    
    // Likelihood
    for (m in 1:M){
        for (c in 1:C){
            k[m,c] ~ binomial_logit(n[m,c], x[m,c]);
        }
    }
    
}
