
//Updated on 5th of April 2024
//@author: Dragos Gruia


data {
  int<lower=1> N;
  vector[N] summary_score;
  vector[N] age;
  vector[N] sex;
  vector[N] education_Alevels;
  vector[N] education_bachelors;
  vector[N] education_postBachelors;
  vector[N] device_phone;
  vector[N] device_tablet;
  vector[N] english_secondLanguage;
  vector[N] depression;
  vector[N] anxiety;
  vector[N] dyslexia;
  
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  
  real<lower=0> sigma;

  real intercept;
  real beta_age;  
  real beta_sex;
  real beta_education_Alevels;
  real beta_education_bachelors;
  real beta_education_postBachelors;
  real beta_device_phone;
  real beta_device_tablet;
  real beta_english_secondLanguage;
  real beta_depression;
  real beta_anxiety;
  real beta_dyslexia;
}
transformed parameters {
  
  // transformed parameters go here
  
  vector[N] mu;
  
  for (i in 1:N)
    mu[i] = intercept + beta_age * age[i] + beta_sex * sex[i] + beta_education_Alevels * education_Alevels[i] + 
      beta_education_bachelors * education_bachelors[i] + beta_education_postBachelors * education_postBachelors[i] + 
      beta_device_phone * device_phone[i] + beta_device_tablet * device_tablet[i] + beta_english_secondLanguage * english_secondLanguage[i] +
      beta_depression * depression[i] + beta_anxiety * anxiety[i] + beta_dyslexia * dyslexia[i];

}


// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.

model {
  
  summary_score ~ normal(mu, sigma);
  
  // Priors go here
    
  intercept ~ normal(0, 1.5);
  beta_age ~ normal(0, 1.5);
  beta_sex ~ normal(0, 1.5);
  beta_education_Alevels ~ normal(0,1.5);
  beta_education_bachelors ~ normal(0,1.5);
  beta_education_postBachelors ~ normal(0,1.5);
  beta_device_phone ~ normal(0,1.5);
  beta_device_tablet ~ normal(0,1.5);
  beta_english_secondLanguage ~ normal(0,1.5);
  beta_depression ~ normal(0,1.5);
  beta_anxiety ~ normal(0,1.5);
  beta_dyslexia ~ normal(0,1.5);
  sigma ~ exponential(1);
}

generated quantities{
  vector[N] log_lik;
  //vector[N] post_pred;
  
  for(i in 1:N)
    log_lik[i] = normal_lpdf(summary_score[i] | mu[i], sigma);
    //post_pred[i] = normal_rng(mu, sigma);
}

