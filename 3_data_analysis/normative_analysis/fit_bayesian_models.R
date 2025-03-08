
# Updated on 5th of June 2024
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
library(rstan)
library(posterior)
library(rstanarm)
library(bayestestR)
library(see)
library(readxl)

###################

#SETTING UP

###################

options(mc.cores = 12)

df = read.csv('ic3_healthy_cleaned_cog_and_demographics.csv')
df_speech = read.csv('data_summaryScore_speech_normative.csv')

df_speech %<>% 
  rename(english_secondLanguage = english) %>%
  mutate(user_id = if_else(user_id =="", ID, user_id))



#df_speech = df_speech[,c('naming','n_events_naming', 'reading','n_events_reading', 'repetition', 'n_events_repetition','user_id')]
df = full_join(df, df_speech, by = c("user_id","age",'gender','english_secondLanguage','education_Alevels','education_bachelors','education_postBachelors','device_phone','device_tablet'))


#x_test = read.csv('IC3_summaryScores_demographics_patients.csv')
x_test = read_excel('summary_cognition_and_demographics.xlsx')
x_test_speech = read.csv('IC3_speechClean_patients.csv')

x_test = left_join(x_test, x_test_speech, by='user_id')

mean_controls = mean(df$age)
std_controls = sd(df$age)

x_test %<>% 
  mutate(
    age = (age - mean_controls)/std_controls
  )

df %<>% 
  mutate(
    age = (age -  mean(df$age))/sd(df$age)
  )

x_test_clean = x_test
df_clean = df

vars_to_keep = c('age', 'gender', 'education_Alevels','education_bachelors', 'education_postBachelors', 'device_phone','device_tablet','english_secondLanguage','depression','anxiety','dyslexia')
vars_to_keep_speech = c('age', 'gender', 'education_Alevels','education_bachelors', 'education_postBachelors', 'device_phone','device_tablet','english_secondLanguage')

dev_from_expected = x_test_clean['user_id']
dev_from_expected = dev_from_expected %>% distinct()

linTaskList = c('blocks','srt','pal','digitSpan','oddOneOut','spatialSpan')


###################

# DEVIATION FROM EXPECTED FOR LINEAR MODELS

###################

for (task in linTaskList) {
  
  df$summary_score = df[, task] #updates each time
  x_test$summary_score = x_test[, task]
  
  df = df[is.na(df$summary_score) == FALSE,]  
  x_test = x_test[is.na(x_test$summary_score) == FALSE,] 
  users_temp = x_test$user_id
  
  real_estimates = x_test$summary_score
  real_estimates = (x_test$summary_score - mean(df$summary_score,na.rm=TRUE))/sd(df$summary_score,na.rm=TRUE)
  
  x_test = x_test[ , vars_to_keep]
  x_test = x_test[ , order(names(x_test))]

  path_to_task = sprintf("models_normative/model_fit/%s.Rda", task)
  stan_save = here::here(path_to_task) 
  load(stan_save)
  
  #Prepare summary statistics for output
  stan_obj_mixed |>
    as_mcmc.list() %>%
    do.call(rbind, .) |>
    as.data.frame() |>
    select(
      matches(c("beta",'intercept'),)
    ) -> posterior_samples
  
  posterior_samples = posterior_samples[,order(names(posterior_samples))]
  
  stan_obj_mixed |>
    as_mcmc.list() %>%
    do.call(rbind, .) |>
    as.data.frame() |>
    select(
      matches(c("sigma"),)
    ) -> sigma_posterior
  
  for_stan = list(
    N = dim(x_test)[1],
    N_vars = dim(x_test)[2],
    N_samples = dim(posterior_samples)[1],
    summary_score = df$summary_score,
    x_test = x_test,
    posterior_samples = posterior_samples,
    sigma_posterior = sigma_posterior[,1]
  )
  
  pred <- stan(file = "models_normative/predict_linear.stan",
               data = for_stan,
               chains = 1, iter = 50,
               algorithm = "Fixed_param")
  
  
  ext_pred <- extract(pred, pars = c('y_test'))
  
  dfe = colMeans(ext_pred$y_test, na.rm = TRUE)
  dfe = apply(dfe, 2, median) %>% round(2) 
  dfe = real_estimates - dfe 
  dfe = data.frame(users_temp,dfe)
  
  dev_from_expected = left_join(dev_from_expected, dfe, by= c("user_id" = "users_temp"))
  names(dev_from_expected)[names(dev_from_expected)=='dfe'] = task
  
  df = df_clean
  x_test = x_test_clean
  
  print(task)
}

write.csv(dev_from_expected, 'linear_models.csv')

#############################

# DEVIATION FROM EXPECTED FOR  BINOMIAL MODELS

#############################


dev_from_expected = read.csv('linear_models.csv')

dev_from_expected = dev_from_expected %>% distinct()

binTaskList = c('trailAll', 'crt','motorControl','naming','reading','repetition', 'auditoryAttention','comprehension','gesture','orientation','pear','semantics','taskRecall','calculation','ided')

for (task in binTaskList) {
  
  df$summary_score = df[, task] # Set summary score as the current task on train
  x_test$summary_score = x_test[, task] # Set summary score as the current task on test
  
  df = df[is.na(df$summary_score) == FALSE,]  # Remove subjects that do not have the summary score
  x_test = x_test[is.na(x_test$summary_score) == FALSE,] # Remove subjects that do not have the summary score
  users_temp = x_test$user_id # Save the users that have done this task
  
  real_estimates = x_test$summary_score
  real_estimates = (x_test$summary_score - mean(df$summary_score,na.rm=TRUE))/sd(df$summary_score,na.rm=TRUE)
  
  if (any(task == c('naming','reading','repetition'))){
    
    col_events = paste('n_events',task,sep="_")
    n_events = x_test[col_events]
    n_events = as.numeric(unlist(n_events))

    x_test = x_test[ , vars_to_keep_speech] 
    x_test = x_test[ , order(names(x_test))]
    
  } else {
    
    n_events = max(df$summary_score)
    n_events <- rep(n_events,times=nrow(x_test))
    x_test = x_test[ , vars_to_keep] 
    x_test = x_test[ , order(names(x_test))]
    
  }
  
  path_to_task = sprintf("models_normative/model_fit/%s.Rda", task)
  stan_save = here::here(path_to_task) 
  load(stan_save)
  
  #Prepare summary statistics for output
  stan_obj_mixed |>
    as_mcmc.list() %>%
    do.call(rbind, .) |>
    as.data.frame() |>
    select(
      matches(c("beta",'intercept'),)
    ) -> posterior_samples
  
  posterior_samples = posterior_samples[,order(names(posterior_samples))]
  
  for_stan = list(
    N = dim(x_test)[1],
    N_vars = dim(x_test)[2],
    N_samples = dim(posterior_samples)[1],
    summary_score = as.integer(df$summary_score),
    x_test = x_test,
    posterior_samples = posterior_samples,
    N_questions = n_events
  )
  
  if (any(task == c('naming','reading','repetition'))){
    pred <- stan(file = "models_normative/predict_binomial_speech.stan",
                 data = for_stan,
                 chains = 1, iter = 50,
                 algorithm = "Fixed_param")
  } else {
    pred <- stan(file = "models_normative/predict_binomial.stan",
                 data = for_stan,
                 chains = 1, iter = 50,
                 algorithm = "Fixed_param")
  }
  
  ext_pred <- extract(pred, pars = c('y_test'))
  dfe = colMeans(ext_pred$y_test, na.rm = TRUE)
  
  # Compare real estimates to posterior predictions 
  
  dfe = apply(dfe, 2, median) %>% round(2) 
  dfe = (dfe - mean(df$summary_score,na.rm=TRUE))/sd(df$summary_score,na.rm=TRUE) # Standardize Bayesian estimates
  dfe = real_estimates - dfe 
  
  dfe = data.frame(users_temp,dfe)
  dfe = dfe %>% distinct(users_temp, .keep_all = TRUE)
  
  dev_from_expected = left_join(dev_from_expected, dfe, by= c("user_id" = "users_temp"))
  names(dev_from_expected)[names(dev_from_expected)=='dfe'] = task
  
  df = df_clean
  x_test = x_test_clean
  print(task)
}

#df_demographics = subset(x_test, select=c(user_id, age:n_events_reading))
#dev_from_expected2 = left_join(dev_from_expected, df_demographics, by= c("user_id" = "user_id"))

write.csv(dev_from_expected, 'dev_from_expected2.csv')

#########################

### CALLCULATE R SQUARED 

#########################

taskList = c('blocks','srt','pal','digitSpan','oddOneOut','spatialSpan', 'trailAll', 'crt','motorControl','naming','reading','repetition', 'auditoryAttention','comprehension','gesture','orientation','pear','semantics','taskRecall','calculation','ided')
r_squared = c()
for (task in taskList) {
  
  path_to_task = sprintf("models_normative/model_fit/%s.Rda", task)
  stan_save = here::here(path_to_task) 
  load(stan_save)
  
  stan_obj_mixed |>
    as_mcmc.list() %>%
    do.call(rbind, .) |>
    as.data.frame() |>
    select(
      matches(c("mu"),)
    ) -> mu_samples
  
  if (any(task == c('naming','reading','repetition'))){ #check if speech data
    
    df = read.csv('data_summaryScore_speech_normative.csv')
  } else {
    df = read.csv('ic3_healthy_cleaned_cog_and_demographics.csv')
  }
  
  df$summary_score = df[,task] 
  df = df[is.na(df$summary_score) == FALSE,]
  
  if (any(task == c('blocks','srt','pal','digitSpan','oddOneOut','spatialSpan'))){ #check if linear model used

    df %<>% mutate(summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE))
    y = df$summary_score
    
  } else { 
    
    n_events = max(df$summary_score)
    y = df$summary_score / n_events
    
    }

  residuals <- -1 * sweep(mu_samples, 2, y)
  var_ypred <- apply(mu_samples, 1, var)
  var_residuals <- apply(residuals, 1, var)
  r_squared_temp = median(var_ypred / (var_ypred + var_residuals))
  r_squared = c(r_squared, r_squared_temp)
  print(task)
  print(r_squared_temp)
  
}

# Format and output the results

r_df = data.frame(taskList,r_squared)
r_df$r_squared = r_df$r_squared %>% round(3) * 100
r_df

write.csv(r_df, 'r_squared_normativeModels.csv')

min(r_df$r_squared)
max(r_df$r_squared)
mean(r_df$r_squared)






