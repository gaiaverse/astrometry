data {
    int<lower=0> P;                       // number of pixels
    int<lower=0> M;                       // number of bins in magnitude space
    int<lower=0> M_subspace;              // number of inducing points in magnitude space
    int<lower=0> C;                       // number of bins in colour space
    int<lower=0> C_subspace;              // number of inducing points in colour space
    int<lower=0> H;                       // number of harmonics
    matrix[P,H] harmonic_to_pixel;        // spherical harmonics
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
    
    // Loop over magnitude and colour
    for (m in 1:M){
        for (c in 1:C){

            // Local variable
            vector[H] a;
            
            // Compute a 
            for (h in 1:H){
                a[h] = mu[h] + sigma[h] * cholesky_m[m] * z[h] * cholesky_c[c];
            }
            
            // Compute x
            x[m,c] = harmonic_to_pixel * a;
            
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
