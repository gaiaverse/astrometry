functions {
    // return (A \otimes B) v where:
    // A is n1 x n1, B = n2 x n2, V = n2 x n1 = reshape(v,n2,n1)
    matrix kron_mvprod(matrix A, matrix B, matrix V) {
        return transpose(A * transpose(B * V));
    }  
}
data {
    int<lower=0> Nm;      // number of bins in magnitude space
    int<lower=0> Nc;      // number of bins in colour space
    matrix[Nm,Nm] Lm;     // Cholesky factor in magnitude space
    matrix[Nc,Nc] Lc;     // Cholesky factor in colour space
    int k[Nm*Nc];
    int n[Nm*Nc];
    real mu;
    real<lower=0> sigma;
}
parameters {
    matrix[Nm,Nc] z;
}
transformed parameters {
    matrix[Nm,Nc] x;
    x = mu + sigma * kron_mvprod(Lc, Lm, z);
}
model {
    to_vector(z) ~ std_normal();
    k ~ binomial_logit(n, to_vector(x));
}