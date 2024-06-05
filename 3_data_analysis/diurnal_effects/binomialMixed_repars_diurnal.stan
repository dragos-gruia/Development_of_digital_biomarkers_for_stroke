
data {
  int<lower=1> N;
  int<lower=1> N_ids;

  array[N] int<lower=0> n_questions;
  array[N] int<lower=0> summary_score;
  array[N] int<lower=1,upper=N_ids> ids;
  
  vector[N] age;
  vector[N] sex;
  vector[N] session;
  vector[N] time_diff;
  vector[N] time_hours;
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
  real beta_session; 
  real beta_time_hours;
  real beta_time_diff;
  real beta_diff_inter;
  real beta_hours_inter;
  real beta_education_bachelors;
  real beta_education_postBachelors;
  real beta_device_type;
}
transformed parameters {
  vector[N] mu;
  vector[N] ids_alpha;
  
  for (i in 1:N) {
    ids_alpha[i] = z[ids[i]] * sigma_alpha;
    mu[i] = inv_logit(intercept + ids_alpha[i] + beta_age * age[i] + beta_sex * sex[i] + beta_session * session[i] +  beta_time_diff * time_diff[i] + beta_time_hours * time_hours[i] + beta_diff_inter * (time_diff[i].* session[i]) + beta_hours_inter * (time_hours[i].* session[i]) + beta_education_bachelors * education_bachelors[i] + beta_education_postBachelors * education_postBachelors[i] + beta_device_type * device_type[i]);
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
  beta_session ~ normal(0, 1);
  beta_time_hours ~ normal(0, 1);
  beta_time_diff ~ normal(0, 1);
  beta_diff_inter ~ normal(0, 1);
  beta_hours_inter ~ normal(0, 1);
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
