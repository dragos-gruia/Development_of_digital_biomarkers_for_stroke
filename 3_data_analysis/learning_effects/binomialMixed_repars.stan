
// Updated on 5th of April 2024
// @author: Dragos Gruia

data {
  int<lower=1> N;
  int<lower=1> N_ids;
  int<lower=1> N_timepoints;

  array[N] int<lower=0> n_questions;
  array[N] int<lower=0> summary_score;
  array[N] int<lower=1,upper=N_ids> ids;
  
  vector[N] age;
  vector[N] sex;
  array[N] int<lower=1> timepoint;
  vector[N] education_bachelors;
  vector[N] education_postBachelors;
  vector[N] device_type;

}


parameters {
  
  real<lower=0> sigma_alpha;
  
  vector[N_ids] z;

  real intercept;
  real beta_age;  
  real beta_sex; 
  vector[N_timepoints] beta_timepoint; 
  real beta_education_bachelors;
  real beta_education_postBachelors;
  real beta_device_type;
}
transformed parameters {
  // transformed parameters go here
  vector[N] mu;
  vector[N] ids_alpha;
  
  for (i in 1:N) {
    ids_alpha[i] = z[ids[i]] * sigma_alpha;
    mu[i] = inv_logit(intercept + ids_alpha[i] + beta_age * age[i] + beta_sex * sex[i] + beta_timepoint[timepoint[i]] + beta_education_bachelors * education_bachelors[i] + beta_education_postBachelors * education_postBachelors[i] + beta_device_type * device_type[i]);
  }
}
model {
  
  // 2nd level
  
  sigma_alpha ~ exponential(1);
  z ~ std_normal(); 
  
  // 1st Level
    
  intercept ~ normal(0.5, 1);
  beta_age ~ normal(0, 1);
  beta_sex ~ normal(0, 1);
  beta_timepoint ~ normal(0, 1);
  beta_education_bachelors ~ normal(0,1);
  beta_education_postBachelors ~ normal(0,1);
  beta_device_type ~ normal(0,1);
  
   for (i in 1:N) {
    summary_score[i] ~ binomial(n_questions[i], mu[i]);
   }
  
}

generated quantities{
  vector[N] log_lik;
  //vector[N] post_pred;
  
  
  for(i in 1:N)
    log_lik[i] = binomial_lpmf(summary_score[i] | n_questions[i], mu[i]);
    //post_pred[i] = normal_rng(mu, sigma);
}
