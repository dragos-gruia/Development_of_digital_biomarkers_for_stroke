

//Updated on 5th of April 2024
//@author: Dragos Gruia

data {
  int<lower=0> N;
  int<lower=0> N_samples; #number of samples in your posterior distribution
  int<lower=0> N_vars;
  
  matrix[N, N_vars] x_test; #the test-set you want to predict
  matrix[N_samples,N_vars+1] posterior_samples;
  array[N] int<lower=0> N_questions; #samples from the posterior distribution
}


parameters {
}

model {
}

generated quantities {
  matrix[N_samples, N] y_test;
  real mu;
  
  for(n in 1:N) {
    
    for(i in 1:N_samples) {
      
      mu = inv_logit(posterior_samples[i,1] * x_test[n,1] + posterior_samples[i,2] * x_test[n,3] + posterior_samples[i,3] * x_test[n,3] + posterior_samples[i,4] * x_test[n,4] +
      posterior_samples[i,5] * x_test[n,5] + posterior_samples[i,6] * x_test[n,6] + posterior_samples[i,7] * x_test[n,7] + posterior_samples[i,8] * x_test[n,8]
      + posterior_samples[i,9]);
      
      y_test[i,n] = binomial_rng(N_questions[n], mu);     
      
    }  
  }
}

