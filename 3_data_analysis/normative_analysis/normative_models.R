
# Updated on 5th of April 2024
# @author: Dragos Gruia

library(here)
library(dplyr)
library(tidyr)
library(magrittr)
library(cmdstanr)
library(shinystan)
library(parallel)
library(stats)
library(viridis)
library(bayesplot)
library(ggplot2)
library(rstan)
library(posterior)
library(rstanarm)
library(bayestestR)
library(see)

df = read.csv('ic3_healthy_cleaned_cog_and_demographics.csv')

df %<>% 
  mutate(
    age= (age - mean(age,na.rm=TRUE))/sd(age,na.rm=TRUE)
  )

df_clean = df
options(mc.cores = 8)

################################# 
#################################  ANALYSE DATA FOR ORIENTATION
################################# 

df$summary_score = df$orientation
df = df[is.na(df$summary_score) == FALSE,]

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))

df <- cbind(df,n_events)

#Frequentist version for comparison

library(lme4)
model <- glm(cbind(summary_score,n_events) ~ age + device_phone + device_tablet + gender + education_Alevels + education_bachelors + education_postBachelors, data=df,
               family=binomial(link="logit"))
summary(model)


for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia,
  n_questions = as.integer(n_events)
)


## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple.stan")
stan_save = here::here("models_normative/model_fit/orientation.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

stan_obj_mixed = load('models_normative/linearMultiple.Rda')

loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 
loo_obj_rslopes = stan_obj_mixed_rslopes$loo(save_psis = TRUE) 

loo::loo_compare(loo_obj,loo_obj_rslopes)

stan_obj_mixed = stan_obj_mixed_rslopes 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta","sigma",'intercept'),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('Orientation'), 
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR Task Recall
################################# 

df$summary_score = df$taskRecall 
df = df[is.na(df$summary_score) == FALSE,]

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia,
  n_questions = as.integer(n_events)
)

#Frequentist version for comparison

library(lme4)

model <- glm(cbind(summary_score,n_events) ~ age + device_phone + device_tablet + sex + education_Alevels + education_bachelors + education_postBachelors, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple.stan")
stan_save = here::here("models_normative/model_fit/taskRecall.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

stan_obj_mixed = stan_obj_mixed_rslopes 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta","sigma"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('Task Recall'), 
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR PAL
################################# 

df$summary_score = df$pal
df = df[is.na(df$summary_score) == FALSE,]

hist(df$summary_score)

df %<>% 
  mutate(
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE)
  )

for_stan = list(
  N = nrow(df),
  summary_score = df$summary_score,
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia
)

hist(for_stan$summary_score)

## Run Bayesian Regression 

stan_file = here::here("models_normative/linearMultiple.stan")
stan_save = here::here("models_normative/model_fit/pal.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 500 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)


loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('PAL'),
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean

################################# 
#################################  ANALYSE DATA FOR DIGITS SPAN 
################################# 

df$summary_score = df$digitSpan 
df = df[is.na(df$summary_score) == FALSE,]


df %<>% 
  mutate(
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE)
  )

for_stan = list(
  N = nrow(df),
  summary_score = df$summary_score,
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia
)

## Run Bayesian Regression 

stan_file = here::here("models_normative/linearMultiple.stan")
stan_save = here::here("models_normative/model_fit/digitSpan.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta","sigma",'intercept'),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars

df_results <- data.frame(Task = c('Digits Span'), 
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR SPATIAL SPAN
################################# 

df$summary_score = df$spatialSpan
df = df[is.na(df$summary_score) == FALSE,]

df %<>% 
  mutate(
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE)
  )

for_stan = list(
  N = nrow(df),
  summary_score = df$summary_score,
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia
)

hist(for_stan$summary_score)

## Run Bayesian Regression 

stan_file = here::here("models_normative/linearMultiple.stan")
stan_save = here::here("models_normative/model_fit/spatialSpan.Rda")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta","sigma"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars

df_results <- data.frame(Task = c('Spatial Span'),
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR COMPREHENSION
################################# 

df$summary_score = df$comprehension 
df = df[is.na(df$summary_score) == FALSE,]


n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia,
  n_questions = as.integer(n_events)
)

hist(for_stan$summary_score)

library(lme4)

model <- glm(cbind(summary_score,n_events) ~ age + device_phone + device_tablet + sex + education_Alevels + education_bachelors + education_postBachelors, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple.stan")
stan_save = here::here("models_normative/model_fit/comprehension.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics

summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('Comprehension'),
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR SEMANTICS
################################# 

df$summary_score = df$semantics
df = df[is.na(df$summary_score) == FALSE,]

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia,
  n_questions = as.integer(n_events)
)

hist(for_stan$summary_score)

library(lme4)

model <- glm(cbind(summary_score,n_events) ~ age + device_phone + device_tablet + sex + education_Alevels + education_bachelors + education_postBachelors, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple.stan")
stan_save = here::here("models_normative/model_fit/semantics.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))

#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('Semantics'), 
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean


################################################################################

#### SPEECH DATA 

################################################################################


df = read.csv('data_summaryScore_speech_normative.csv')
df %<>% 
  mutate(user_id = coalesce(ID,user_id))

df_clean = df
options(mc.cores = 8)


################################# 
#################################  ANALYSE DATA FOR NAMING TASK
################################# 

df$summary_score = df$naming 
df = df[is.na(df$summary_score) == FALSE,]
df = df[is.na(df$gender) == FALSE,]


df %<>% 
  mutate(
    age = (age - mean(age,na.rm=TRUE))/sd(age,na.rm=TRUE),
    ID = as.integer(as.factor(ID))
  )


for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english,
  n_questions = as.integer(df$n_events_naming)
)

hist(df$summary_score/df$n_events_naming)
hist(df$summary_score)

library(lme4)

model <- glm(cbind(summary_score,n_questions) ~ age + device_phone + device_tablet + sex + education_Alevels + education_bachelors + education_postBachelors + english_secondLanguage, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple_speech.stan")
stan_save = here::here("models_normative/model_fit/naming.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, 
  prob_outer = 0.95, # 99%
  point_est = "mean"
) +
  labs(
  title = "Naming Task",
  subtitle = "With median beta coeffcients and 95% intervals"
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars

round(apply(scale_pars,2,mean),2)

save(stan_obj_mixed, file = stan_save)

df = df_clean

################################# 
#################################  ANALYSE DATA FOR READING TASK
################################# 

df$summary_score = df$reading 
df = df[is.na(df$summary_score) == FALSE,]
df = df[is.na(df$gender) == FALSE,]

df %<>% 
  mutate(
    age = (age - mean(age,na.rm=TRUE))/sd(age,na.rm=TRUE),
    ID = as.integer(as.factor(ID))
  )

for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english,
  n_questions = as.integer(df$n_events_reading)
)

hist(for_stan$summary_score)

library(lme4)

model <- glm(cbind(summary_score,n_questions) ~ age + device_phone + device_tablet + sex + education_Alevels + education_bachelors + education_postBachelors + english_secondLanguage, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple_speech.stan")
stan_save = here::here("models_normative/model_fit/reading.Rda")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, 
  prob_outer = 0.95, # 99%
  point_est = "mean"
) +
  labs(
    title = "Reading Task",
    subtitle = "With median beta coeffcients and 95% intervals"
  )


#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars

round(apply(scale_pars,2,mean),2)

save(stan_obj_mixed, file = stan_save)

df = df_clean

################################# 
#################################  ANALYSE DATA FOR REPETITION TASK
################################# 

df$summary_score = df$repetition
df = df[is.na(df$summary_score) == FALSE,]
df = df[is.na(df$gender) == FALSE,]

df %<>% 
  mutate(
    age = (age - mean(age,na.rm=TRUE))/sd(age,na.rm=TRUE),
    ID = as.integer(as.factor(ID))
  )


for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english,
  n_questions = as.integer(df$n_events_repetition)
)

hist(for_stan$summary_score)

library(lme4)

model <- glm(cbind(summary_score,n_questions) ~ age + device_phone + device_tablet + sex + education_Alevels + education_bachelors + education_postBachelors + english_secondLanguage, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple_speech.stan")
stan_save = here::here("models_normative/model_fit/repetition.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)


#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, 
  prob_outer = 0.95, # 99%
  point_est = "mean"
) +
  labs(
    title = "Repetition Task",
    subtitle = "With median beta coeffcients and 95% intervals"
  )


#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars

round(apply(scale_pars,2,mean),2)

save(stan_obj_mixed, file = stan_save)




################################# 
#################################  SWITCH BACK TO NON-SPEECH TASKS
################################# 

df = read.csv('ic3_healthy_cleaned_cog_and_demographics.csv')

df %<>% 
  mutate(
    age= (age - mean(age,na.rm=TRUE))/sd(age,na.rm=TRUE)
  )

df_clean = df


################################# 
#################################  ANALYSE DATA FOR BLOCKS
################################# 

df$summary_score = df$blocks 
df = df[is.na(df$summary_score) == FALSE,]


df %<>% 
  mutate(
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE)
  )

for_stan = list(
  N = nrow(df),
  summary_score = df$summary_score,
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia
)

hist(for_stan$summary_score)

## Run Bayesian Regression 

stan_file = here::here("models_normative/linearMultiple.stan")
stan_save = here::here("models_normative/model_fit/blocks.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta","sigma"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('Blocks'), 
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean

################################# 
#################################  ANALYSE DATA FOR TRAIL
################################# 

df$summary_score = df$trailAll
df = df[is.na(df$summary_score) == FALSE,]


n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia,
  n_questions = as.integer(n_events)
)

hist(for_stan$summary_score)

library(lme4)

model <- glm(cbind(summary_score,n_events) ~ age + device_phone + device_tablet + sex + education_Alevels + education_bachelors + education_postBachelors, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple.stan")
stan_save = here::here("models_normative/model_fit/trailAll.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)


loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('Trail'),
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR ODD ONE OUT
################################# 

df$summary_score = df$oddOneOut 
df = df[is.na(df$summary_score) == FALSE,]

hist(df$summary_score)

df %<>% 
  mutate(
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE)
  )

for_stan = list(
  N = nrow(df),
  summary_score = df$summary_score,
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia
  
)

hist(for_stan$summary_score)

## Run Bayesian Regression 

stan_file = here::here("models_normative/linearMultiple.stan")
stan_save = here::here("models_normative/model_fit/oddOneOut.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)


loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('Odd One Out'), 
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR IDED
################################# 

df$summary_score = df$ided 
df = df[is.na(df$summary_score) == FALSE,]


n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia,
  n_questions = as.integer(n_events)
)

hist(for_stan$summary_score)

library(lme4)

model <- glm(cbind(summary_score,n_events) ~ age + device_phone + device_tablet + sex + education_Alevels + education_bachelors + education_postBachelors, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple.stan")
stan_save = here::here("models_normative/model_fit/ided.Rda")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)


loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('IDED'),
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR PEAR
################################# 

df$summary_score = df$pear
df = df[is.na(df$summary_score) == FALSE,]



df %<>% 
  mutate(
    summary_score = summary_score*20
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia,
  n_questions = as.integer(n_events)
)

library(lme4)
model <- glm(cbind(summary_score,n_events) ~ age + device_phone + device_tablet + sex + education_Alevels + education_bachelors + education_postBachelors, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple.stan")
stan_save = here::here("models_normative/model_fit/pear.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta","sigma",'intercept'),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('Pear Cancellation'),
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR SIMPLE REACTION TIME
################################# 

df$summary_score = df$srt
df = df[is.na(df$summary_score) == FALSE,]

hist(df$summary_score)

df %<>% 
  mutate(
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE)
  )

for_stan = list(
  N = nrow(df),
  summary_score = df$summary_score,
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia
  
)

hist(for_stan$summary_score)

## Run Bayesian Regression 

stan_file = here::here("models_normative/linearMultiple.stan")
stan_save = here::here("models_normative/model_fit/srt.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta","sigma",'intercept'),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('SRT'),
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR AUDITORY ATTENTION
################################# 

df$summary_score = df$auditoryAttention 
df = df[is.na(df$summary_score) == FALSE,]


n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia,
  n_questions = as.integer(n_events)
)

hist(for_stan$summary_score)

library(lme4)

model <- glm(cbind(summary_score,n_events) ~ age + device_phone + device_tablet + sex + education_Alevels + education_bachelors + education_postBachelors, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple.stan")
stan_save = here::here("models_normative/model_fit/auditoryAttention.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta","sigma"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('Auditory Attention'),
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean



################################# 
#################################  ANALYSE DATA FOR CHOICE REACTION TIME
################################# 

df$summary_score = df$crt
df = df[is.na(df$summary_score) == FALSE,]

hist(df$summary_score)

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia,
  n_questions = as.integer(n_events)
)

hist(for_stan$summary_score)

library(lme4)

model <- glm(cbind(summary_score,n_events) ~ age + device_phone + device_tablet + sex + education_Alevels + education_bachelors + education_postBachelors, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple.stan")
stan_save = here::here("models_normative/model_fit/crt.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('CRT'),
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR MOTOR CONTROL
################################# 

df$summary_score = df$motorControl
df = df[is.na(df$summary_score) == FALSE,]


n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia,
  n_questions = as.integer(n_events)
)

hist(for_stan$summary_score)

library(lme4)

model <- glm(cbind(summary_score,n_events) ~ age + device_phone + device_tablet + sex + education_Alevels + education_bachelors + education_postBachelors, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple.stan")
stan_save = here::here("models_normative/model_fit/motorControl.Rda")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('Trail'),
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean



################################# 
#################################  ANALYSE DATA FOR CALCULATION
################################# 

df$summary_score = df$calculation 
df = df[is.na(df$summary_score) == FALSE,]


n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia,
  n_questions = as.integer(n_events)
)

hist(for_stan$summary_score)

library(lme4)


model <- lm(summary_score ~ age + device_phone + device_tablet + gender + education_Alevels + education_bachelors + education_postBachelors + english_secondLanguage + depression + anxiety + dyslexia, data=df)
summary(model)             

model <- glm(cbind(summary_score,n_events) ~ age + sex + education_Alevels + education_bachelors + education_postBachelors + device_phone + device_tablet + english_secondLanguage + depression + anxiety + dyslexia, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple.stan")
stan_save = here::here("models_normative/model_fit/calculation.Rda") 

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 

y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('Calculation'), 
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean



################################# 
#################################  ANALYSE DATA FOR GESTURE
################################# 

df$summary_score = df$gesture 
df = df[is.na(df$summary_score) == FALSE,]

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = nrow(df),
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,    
  education_Alevels = df$education_Alevels,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  device_phone = df$device_phone,
  device_tablet = df$device_tablet,
  english_secondLanguage = df$english_secondLanguage,
  depression = df$depression,
  anxiety = df$anxiety,
  dyslexia = df$dyslexia,
  n_questions = as.integer(n_events)
)

hist(for_stan$summary_score)

library(lme4)

model <- glm(cbind(summary_score,n_events) ~ age + device_phone + device_tablet + sex + education_Alevels + education_bachelors + education_postBachelors, data=for_stan,
             family=binomial(link="logit"))
summary(model)

## Run Bayesian Regression 

stan_file = here::here("models_normative/binomialMultiple.stan")
stan_save = here::here("models_normative/model_fit/gesture.Rda")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 13,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

save(stan_obj_mixed, file = stan_save)
#load(stan_save)

loo_obj = stan_obj_mixed$loo(save_psis = TRUE) 
y0 = for_stan$summary_score

#Print diagnostics
summaries = stan_obj_mixed$summary()
keep_params <- c("intercept","beta_age","beta_sex","beta_education_bachelors","beta_education_postBachelors","beta_device_phone","sigma")
params_summary <- summaries[sapply(summaries$variable, function(x) any(x == keep_params)),]
print(params_summary)

stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)

#Plot betas
bayesplot::mcmc_hist(stan_obj_mixed$draws(), 
                     pars = vars(matches(c("beta","sigma",'intercept'),))
)

#Plot MCMC traces
color_scheme_set("mix-blue-red")
bayesplot::mcmc_trace(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))


#Check auto-correlation
color_scheme_set("purple")
p<- mcmc_acf(stan_obj_mixed$draws(), pars = vars(matches(c("beta","sigma",'intercept'),)))
#p<- mcmc_acf_bar(stan_obj_mixed$draws(), pars = c("sigma_alpha"))
p + hline_at(0.5, linetype = 2, size = 0.15, color = "black")

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),)
  ) -> scale_pars


df_results <- data.frame(Task = c('Gesture'),
                         N = c(for_stan$N_ids),
                         beta_standardised = c(mean(scale_pars$beta_session)), 
                         p_value = sum(scale_pars$beta_session <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$beta_session,ci=1)))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean

