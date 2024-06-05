
// Updated on 5th of April 2024
// @author: Dragos Gruia

data {
  int<lower=1> N;
  int<lower=1> N_ids;
  int<lower=1> N_timepoints;
  
  array[N] int<lower=1> timepoint;
  array[N] int<lower=1,upper=N_ids> ids;
  
  vector[N] summary_score;
  vector[N] age;
  vector[N] sex;

  vector[N] time;
  vector[N] education_bachelors;
  vector[N] education_postBachelors;
  vector[N] device_type;
  
}

parameters {
  
  real<lower=0> sigma;
  real<lower=0> sigma_alpha;
  
  vector[N_timepoints] beta_timepoint;
  vector[N_ids] z;

  real intercept;
  real beta_age;  
  real beta_sex; 
  real beta_time;
  real beta_education_bachelors;
  real beta_education_postBachelors;
  real beta_device_type;

}
transformed parameters {
  // transformed parameters go here
  vector[N] mu;
  real alpha_rep;
  
  for (i in 1:N) {
    alpha_rep = sigma_alpha * z[ids[i]];
    mu[i] = intercept + alpha_rep + beta_age * age[i] + beta_sex * sex[i] + beta_timepoint[timepoint[i]] + beta_time * time[i] + beta_education_bachelors * education_bachelors[i] + beta_education_postBachelors * education_postBachelors[i] + beta_device_type * device_type[i];
    }
}

model {

  // 2nd level
  
  z ~ std_normal();
  sigma_alpha ~ exponential(1);
  
  // 1st level
    
  intercept ~ normal(0, 1.5);
  beta_age ~ normal(0, 1.5);
  beta_sex ~ normal(0, 1.5);
  beta_timepoint ~ normal(0, 1.5);
  beta_time ~ normal(0, 1.5);
  beta_education_bachelors ~ normal(0,1.5);
  beta_education_postBachelors ~ normal(0,1.5);
  beta_device_type ~ normal(0,1.5);
  sigma ~ exponential(1);
  
  summary_score ~ normal(mu, sigma);
  
}

generated quantities{
  vector[N] log_lik;
  //vector[N] post_pred;
  
  
  for(i in 1:N)
    log_lik[i] = normal_lpdf(summary_score[i] | mu[i], sigma);
    //post_pred[i] = normal_rng(mu, sigma);
}

