
//Updated on 5th of April 2024
//@author: Dragos Gruia

data {
  int<lower=1> N;

  vector[N] age;
  vector[N] sex;
  vector[N] education_Alevels;
  vector[N] education_bachelors;
  vector[N] education_postBachelors;
  vector[N] device_phone;
  vector[N] device_tablet;
  vector[N] english_secondLanguage;
  
  array[N] int<lower=0> n_questions;
  array[N] int<lower=0> summary_score;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'N'.

parameters {

  real intercept;
  real beta_age;  
  real beta_sex;
  real beta_education_Alevels;
  real beta_education_bachelors;
  real beta_education_postBachelors;
  real beta_device_phone;
  real beta_device_tablet;
  real beta_english_secondLanguage;
}
transformed parameters {
  // transformed parameters go here
  vector[N] mu;

    mu = inv_logit(intercept + beta_age * age + beta_sex * sex + beta_education_Alevels * education_Alevels + 
      beta_education_bachelors * education_bachelors + beta_education_postBachelors * education_postBachelors + 
      beta_device_phone * device_phone + beta_device_tablet * device_tablet + beta_english_secondLanguage * english_secondLanguage);

}
model {
  
  // Likelihood
  
  summary_score ~ binomial(n_questions, mu);

  // Priors go here
    
  intercept ~ normal(0.5, 1);
  beta_age ~ normal(0, 1);
  beta_sex ~ normal(0, 1);
  beta_education_Alevels ~ normal(0,1);
  beta_education_bachelors ~ normal(0,1);
  beta_education_postBachelors ~ normal(0,1);
  beta_device_phone ~ normal(0,1);
  beta_device_tablet ~ normal(0,1);
  beta_english_secondLanguage ~ normal(0,1);
  
  
}

generated quantities{
    vector[N] log_lik;
  //vector[N] post_pred;
  
  for(i in 1:N)
    log_lik[i] = binomial_lpmf(summary_score[i] | n_questions[i], mu[i]);
    //post_pred[i] = normal_rng(mu, sigma);
}

