
# ============================================================================ #
# Script: Deviation from Expected Estimation and R Squared Calculation 
#
#  Last updated on: 5th of June 2024                                         
#  Author: Dragos Gruia   
#
# Description:
#    This script computes deviation from expected scores for patients
#    and calculates R-squared values for the normative models. It uses 
#    previously fitted Bayesian models (both linear and binomial)    
#    to predict expected scores for patient cohorts based on demographics 
#    and clinical variables. Deviations are computed for a range of tasks 
#    using separate approaches for linear and binomial outcomes, and the   
#    script also estimates R-squared for each task model.                
#                                                            
# Inputs:                                                        
#    - ic3_healthy_cleaned_cog_and_demographics.csv                      
#         Normative (control) dataset with cognitive and demographic data
#    - data_summaryScore_speech_normative.csv                            
#         Normative speech dataset (summary scores for speech tasks)     
#    - summary_cognition_and_demographics.xlsx                           
#         Patient cohort data for cognitive performance                   
#    - IC3_speechClean_patients.csv                                      
#         Patient speech data    
#    - Stand models required to                                      
#         Patient speech data    
#
# Outputs:                                                       
#    - linear_models.csv                                                 
#         CSV file with deviations from expected scores for linear tasks 
#    - dev_from_expected.csv                                             
#         CSV file with deviations for binomial tasks                     
#    - r_squared_normativeModels.csv                                     
#         CSV file summarizing R-squared values (in percentages) for each task 
#                                                                      
# ============================================================================ #


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


########## SETTING UP ----------------------------------------------------------


options(mc.cores = 12)

vars_to_keep = c('age', 'gender', 'education_Alevels','education_bachelors',
                 'education_postBachelors', 'device_phone','device_tablet',
                 'english_secondLanguage','depression','anxiety','dyslexia')
vars_to_keep_speech = c('age', 'gender', 'education_Alevels','education_bachelors',
                        'education_postBachelors', 'device_phone','device_tablet',
                        'english_secondLanguage')

# Create the training dataset using the normative cohort
df = read.csv('ic3_healthy_cleaned_cog_and_demographics.csv')
df_speech = read.csv('data_summaryScore_speech_normative.csv')

df_speech %<>% 
  rename(english_secondLanguage = english) %>%
  mutate(user_id = if_else(user_id =="", ID, user_id))

df = full_join(df, df_speech,
               by = c("user_id","age",'gender','english_secondLanguage',
                      'education_Alevels','education_bachelors',
                      'education_postBachelors','device_phone','device_tablet'))

# Create the testing dataset using the patients cohort
x_test = read_excel('summary_cognition_and_demographics.xlsx')
x_test_speech = read.csv('IC3_speechClean_patients.csv')
x_test = left_join(x_test, x_test_speech, by='user_id')

# Standardise age in patients according to the age of controls
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


# Save a copy of the data to use later
x_test_clean = x_test
df_clean = df

# Create a list of patients for which you will create deviation from expected scores
dev_from_expected = x_test_clean['user_id']
dev_from_expected = dev_from_expected %>% distinct()

linTaskList = c('blocks','srt','pal','digitSpan','oddOneOut','spatialSpan')


######################## DEVIATION FROM EXPECTED FOR LINEAR MODELS -------------

for (task in linTaskList) {
  
  # Select only participants who have data on that given task
  df$summary_score = df[, task] #updates each time
  x_test$summary_score = x_test[, task]
  
  df = df[is.na(df$summary_score) == FALSE,]  
  x_test = x_test[is.na(x_test$summary_score) == FALSE,] 
  users_temp = x_test$user_id
  
  # Standardise and save the real cognitive performance
  real_estimates = x_test$summary_score
  real_estimates = (x_test$summary_score - mean(df$summary_score,na.rm=TRUE))/sd(df$summary_score,na.rm=TRUE)
  
  # Save the demographic and clinical variables associated with each patient
  x_test = x_test[ , vars_to_keep]
  x_test = x_test[ , order(names(x_test))]

  # Load the previously computed models
  path_to_task = sprintf("models_normative/model_fit/%s.Rda", task)
  stan_save = here::here(path_to_task) 
  load(stan_save)
  
  # Extract previously computed posterios samples
  stan_obj_mixed |>
    as_mcmc.list() %>%
    do.call(rbind, .) |>
    as.data.frame() |>
    select(
      matches(c("beta",'intercept'),)
    ) -> posterior_samples
  
  posterior_samples = posterior_samples[,order(names(posterior_samples))]
  
  # Extract the SD of the model
  stan_obj_mixed |>
    as_mcmc.list() %>%
    do.call(rbind, .) |>
    as.data.frame() |>
    select(
      matches(c("sigma"),)
    ) -> sigma_posterior
  
  # Create a list of variables to be inputted in stan
  for_stan = list(
    N = dim(x_test)[1],
    N_vars = dim(x_test)[2],
    N_samples = dim(posterior_samples)[1],
    summary_score = df$summary_score,
    x_test = x_test,
    posterior_samples = posterior_samples,
    sigma_posterior = sigma_posterior[,1]
  )
  
  # Predict expected patient scores using their demographics
  # and by using the models computed on healthy adults
  pred <- stan(file = "models_normative/predict_linear.stan",
               data = for_stan,
               chains = 1, iter = 50,
               algorithm = "Fixed_param")
  ext_pred <- extract(pred, pars = c('y_test'))
  
  # Calculate deviation from expected by comparing the predicted score with the patients' real scores
  dfe = colMeans(ext_pred$y_test, na.rm = TRUE)
  dfe = apply(dfe, 2, median) %>% round(2) 
  dfe = real_estimates - dfe 
  dfe = data.frame(users_temp,dfe)
  
  # Add the scores to a dataframe
  dev_from_expected = left_join(dev_from_expected, dfe, by= c("user_id" = "users_temp"))
  names(dev_from_expected)[names(dev_from_expected)=='dfe'] = task
  
  # Reset the dataframe
  df = df_clean
  x_test = x_test_clean
  
  print(task)
}

write.csv(dev_from_expected, 'linear_models.csv')


################## DEVIATION FROM EXPECTED FOR  BINOMIAL MODELS ----------------


dev_from_expected = read.csv('linear_models.csv')
dev_from_expected = dev_from_expected %>% distinct()

binTaskList = c('trailAll', 'crt','motorControl','naming','reading','repetition',
                'auditoryAttention','comprehension','gesture','orientation','pear',
                'semantics','taskRecall','calculation','ided')

for (task in binTaskList) {
  
  # Select only participants who have data on that given task
  df$summary_score = df[, task] 
  x_test$summary_score = x_test[, task] 
  
  df = df[is.na(df$summary_score) == FALSE,]  
  x_test = x_test[is.na(x_test$summary_score) == FALSE,] 
  users_temp = x_test$user_id 
  
  # Standardise and save the real cognitive performance
  real_estimates = x_test$summary_score
  real_estimates = (x_test$summary_score - mean(df$summary_score,na.rm=TRUE))/sd(df$summary_score,na.rm=TRUE)
  
  # Save the demographic and clinical variables associated with each patient
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
  
  # Load the previously computed models
  path_to_task = sprintf("models_normative/model_fit/%s.Rda", task)
  stan_save = here::here(path_to_task) 
  load(stan_save)
  
  stan_obj_mixed |>
    as_mcmc.list() %>%
    do.call(rbind, .) |>
    as.data.frame() |>
    select(
      matches(c("beta",'intercept'),)
    ) -> posterior_samples
  
  posterior_samples = posterior_samples[,order(names(posterior_samples))]
  
  # Create a list of variables to be inputted in stan
  for_stan = list(
    N = dim(x_test)[1],
    N_vars = dim(x_test)[2],
    N_samples = dim(posterior_samples)[1],
    summary_score = as.integer(df$summary_score),
    x_test = x_test,
    posterior_samples = posterior_samples,
    N_questions = n_events
  )
  
  # Predict expected patient scores using their demographics
  # and by using the models computed on healthy adults
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
  
  # Create deviation from expected by comparing real estimates to posterior predictions 
  dfe = colMeans(ext_pred$y_test, na.rm = TRUE)
  dfe = apply(dfe, 2, median) %>% round(2) 
  dfe = (dfe - mean(df$summary_score,na.rm=TRUE))/sd(df$summary_score,na.rm=TRUE) # Standardize Bayesian estimates
  dfe = real_estimates - dfe 
  dfe = data.frame(users_temp,dfe)
  dfe = dfe %>% distinct(users_temp, .keep_all = TRUE)
  
  # Add the scores to a dataframe
  dev_from_expected = left_join(dev_from_expected, dfe, by= c("user_id" = "users_temp"))
  names(dev_from_expected)[names(dev_from_expected)=='dfe'] = task
  
  # Reset dataframe
  df = df_clean
  x_test = x_test_clean
  print(task)
}

write.csv(dev_from_expected, 'dev_from_expected.csv')


########################## CALLCULATE R SQUARED --------------------------------


taskList = c('blocks','srt','pal','digitSpan','oddOneOut','spatialSpan', 'trailAll',
             'crt','motorControl','naming','reading','repetition', 'auditoryAttention',
             'comprehension','gesture','orientation','pear','semantics',
             'taskRecall','calculation','ided')
r_squared = c()

for (task in taskList) {
  
  # Load task-specific bayesian posterior samples
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
  
  # Save the original summary score for the task
  df$summary_score = df[,task] 
  df = df[is.na(df$summary_score) == FALSE,]
  
  # Standardise if linear model used otherwise divide by number of events
  if (any(task == c('blocks','srt','pal','digitSpan','oddOneOut','spatialSpan'))){ 
    df %<>% mutate(summary_score= (
      summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE))
    y = df$summary_score
  } else { 
    n_events = max(df$summary_score)
    y = df$summary_score / n_events
  }

  # Calculate R-squared
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
print(r_df)

min(r_df$r_squared)
max(r_df$r_squared)
mean(r_df$r_squared)

write.csv(r_df, 'r_squared_normativeModels.csv')





