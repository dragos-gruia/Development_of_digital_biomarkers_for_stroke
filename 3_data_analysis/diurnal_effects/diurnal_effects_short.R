
# ============================================================================ #
# Script: Bayesian Diurnal Analysis
#
#  Last updated on: 4th of May 2024                                         
#  Author: Dragos Gruia   
#
# Description:
#    This script performs Bayesian regression analyses on cognitive    
#    performance data collected at different times of day. It analyzes 
#    three tasks: Orientation, Task Recall, and PAL. The script         
#    pre-processes data, fits a binomial mixed-effects model using Stan,  
#    extracts posterior summaries, and compiles the results.           
#                                                            
#
# Inputs:
#    - data_for_bayesian_analysis.csv                                  
#         Primary dataset containing cognitive performance measures    
#         and associated variables (Age, session, startTime, etc.).     
#    - data_summaryScore_speech_validation.csv                           
#         Secondary dataset for speech validation
#
# Outputs:
#    - df_results:                                                     
#         A data frame compiling summary statistics from the Bayesian  
#         regression analyses for each task. It includes the number    
#         of IDs analyzed, ROPE probabilities, and 95% credible          
#         intervals for various effects (session and interaction terms). 
#
# ============================================================================ #


library(here)
library(dplyr)
library(tidyr)
library(magrittr)
library(cmdstanr)
library(shinystan)
library(parallel)
library("rstanarm")  
library(viridis)
library("bayesplot")
library(rstan)
library(posterior)
library(rstanarm)
library(bayestestR)
library(see)
library(lme4)

# Settings up -------
here::here() 
options(mc.cores = 8) 
models_folder = 'models_diurnal'
output_folder = 'models_diurnal/fit_save'
cog_data = 'data_for_bayesian_analysis.csv'
speech_data = 'data_summaryScore_speech_validation.csv'

# Load cognitive data
df = read.csv(cog_data)

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    session = session - 3/2
    )

df_clean = df


############################ ANALYSE ORIENTATION -------------------------------
 

df$summary_score = df$orientation  

to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]


df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
)


## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model(stan_file)

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 10000,
  iter_sampling = 10000,
  refresh = 1000 # print update every 1000 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars


df_results <- data.frame(Task = c('Orientation'), 
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_results) = NULL


df = df_clean



# ANALYSE TASK RECALL -------


df$summary_score = df$taskRecall   

to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
)

## Run Bayesian Regression 


stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling =  10000,
  refresh = 1000 # print update every 500 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars


df_temp <- data.frame(Task = c('Task Recall'), # 
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean


############################### ANALYSE PAL ------------------------------------

df$summary_score = df$pal  

to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
)


## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling =  10000,
  refresh = 1000 # print update every 500 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars

df_temp <- data.frame(Task = c('PAL'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)


df = df_clean


######################### ANALYSE DIGITS SPAN ----------------------------------


df$summary_score = df$digitSpan  
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]


df %<>% 
  mutate(
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE),
    ID = as.integer(as.factor(ID))
  )

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = df$summary_score,
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID
  
)

## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"linMixed_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 10000,
  refresh = 1000 # print update every 500 iters
)


#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars


df_temp <- data.frame(Task = c('Digits Span'),  
                         N = c(for_stan$N_ids),
                         p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1))),
                         lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                         upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                         p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1))),
                         lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                         upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                         p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1))),
                         lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                         upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
                         
)
rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)


df = df_clean



################################## ANALYSE SPATIAL SPAN ------------------------



df$summary_score = df$spatialSpan  
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE),
    ID = as.integer(as.factor(ID))
  )

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = df$summary_score,
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID
  
)

## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"linMixed_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 10000,
  refresh = 500 # print update every 500 iters
)


#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars



df_temp <- data.frame(Task = c('Spatial Span'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)


df = df_clean



################################ ANALYSE COMPREHENSION -------------------------


df$summary_score = df$comprehension  

to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))


for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
)

## Run Bayesian Regression 


stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model(stan_file)

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 10000,
  refresh = 1000 # print update every 1000 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars

df_temp <- data.frame(Task = c('Comprehension'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean


###################################### ANALYSE SEMANTICS -----------------------

df$summary_score = df$semantics  
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
)

## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))


model_obj = cmdstanr::cmdstan_model(stan_file)

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 10000,
  refresh = 1000 # print update every 500 iters
)


#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars


df_temp <- data.frame(Task = c('Semantic Judgement'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean



############################ PREPARE SPEECH ------------------------------------


df = read.csv(speech_data)
options(mc.cores = 8) 

df %<>% 
  mutate(
    age = (age - mean(age,na.rm=TRUE))/sd(age,na.rm=TRUE),
    session = session - 3/2
  )

df_clean = df


##########################  ANALYSE NAMING -------------------------------------

df$summary_score = df$naming  
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,
  session = df$session,
  time_hours = df$startTime_hours,
  time_diff = df$time_diff2,
  device_type = df$device,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(df$n_events_naming)
)


stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling =  10000,
  refresh = 1000 # print update every 500 iters
)


stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)


#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars

df_temp <- data.frame(Task = c('Naming'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean


 
############################  ANALYSE READING ----------------------------------


df$summary_score = df$reading  
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,
  session = df$session,
  time_hours = df$startTime_hours,
  time_diff = df$time_diff2,
  device_type = df$device,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(df$n_events_reading)
)


stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling =  10000,
  refresh = 1000 
)


stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)


#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars

df_temp <- data.frame(Task = c('Reading'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean



#############################  ANALYSE REPETITION ------------------------------
 

df$summary_score = df$repetition  
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,
  session = df$session,
  time_hours = df$startTime_hours,
  time_diff = df$time_diff2,
  device_type = df$device,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(df$n_events_repetition)
)


stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling =  10000,
  refresh = 1000 
)


stan_obj_mixed$diagnostic_summary()

#Check credible intervals 
mcmc_intervals(
  stan_obj_mixed$draws(), 
  pars = vars(matches(c("beta"),)),
  prob = 0.5, # 80% intervals
  prob_outer = 0.95, # 99%
  point_est = "mean"
)


#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars

df_temp <- data.frame(Task = c('Repetition'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean


############################## ANALYSE BLOCKS ----------------------------------

df = read.csv('blocks_for_bayesianR.csv') #removed those who did not have practice trials in one of the sessions
df$summary_score = df$SummaryScore  
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE),
    session = session - 3/2,
    startTime = (startTime - mean(startTime))/sd(startTime),
    ID = as.integer(as.factor(ID))
  )

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = df$summary_score,
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID
  
)

## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"linMixed_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 10000,
  refresh = 500 # print update every 500 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars


df_temp <- data.frame(Task = c('Blocks'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean




################################## SWITCH TO COGNITIVE DATA --------------------



df = read.csv(cog_data)

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    session = session - 3/2
  )

df_clean = df


################################ ANALYSE TRAIL-MAKING --------------------------


df$summary_score = df$trailAll  

to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )


n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))


for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
)

## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model(stan_file)

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 10000,
  refresh = 1000 # print update every 1000 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars


df_temp <- data.frame(Task = c('Trail-making'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)


df = df_clean


##################### ANALYSE ODD ONE OUT --------------------------------------

df$summary_score = df$oddOneOut  
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]


df %<>% 
  mutate(
    ID = as.integer(as.factor(ID)),
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE)
  )

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = df$summary_score,
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID
  
)

## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"linMixed_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 10000,
  refresh = 500 # print update every 500 iters
)


#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars


df_temp <- data.frame(Task = c('Odd One Out'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean



############################### ANALYSE IDED -----------------------------------

df$summary_score = df$ided  

to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )


n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))


for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
)


## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model(stan_file)

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 10000,
  refresh = 1000 # print update every 1000 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars


df_temp <- data.frame(Task = c('Rule Learning'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)


rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean


############################ ANALYSE PEAR CANCELLATION -------------------------

df$summary_score = df$pear  

to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))


for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
)

## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling =  10000,
  refresh = 1000 # print update every 500 iters
)


#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars


df_temp <- data.frame(Task = c('Pear Cancellation'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean


########################## ANALYSE SIMPLE REACTION TIME ------------------------

df$summary_score = df$srt  
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID)),
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE)
  )

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = df$summary_score,
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID
  
)

## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"linMixed_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 10000,
  refresh = 500 # print update every 500 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars

df_temp <- data.frame(Task = c('SRT'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean


####################### ANALYSE AUDITORY ATTENTION -----------------------------


df$summary_score = df$auditoryAttention  

to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))


for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
)


## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model(stan_file)

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 10000,
  refresh = 1000 # print update every 1000 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars


df_temp <- data.frame(Task = c('Auditory Attention'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean


######################### ANALYSE CHOICE REACTION TIME -------------------------


df$summary_score = df$crt  
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]


df %<>% 
  mutate(
    ID = as.integer(as.factor(ID)),
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE)
  )

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = df$summary_score,
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID
  
)

## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"linMixed_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 10000,
  refresh = 500 # print update every 500 iters
)


#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars


df_temp <- data.frame(Task = c('CRT'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean



############################## ANALYSE MOTOR CONTROL ---------------------------


df$summary_score = df$motorControl  

to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )


n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))


for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
)


## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling =  10000,
  refresh = 1000 # print update every 500 iters
)


#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars

df_temp <- data.frame(Task = c('Motor Control'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)
rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean



############################## ANALYSE CALCULATION -----------------------------


df$summary_score = df$calculation  

to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )


n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))


for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
)

## Run Bayesian Regression 

stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model(stan_file)

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 10000,
  refresh = 1000 # print update every 1000 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars


df_temp <- data.frame(Task = c('Calculation'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean


################################# ANALYSE GESTURE ------------------------------

df$summary_score = df$gesture  

to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

df %<>% 
  mutate(
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))


for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/2,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  session = df$session,
  time_hours = df$startTime,
  time_diff = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
)


## Run Bayesian Regression 


stan_file = here::here(paste(models_folder,"binomialMixed_repars_diurnal.stan",sep='/'))

model_obj = cmdstanr::cmdstan_model(stan_file)

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 5000,
  iter_sampling = 10000,
  refresh = 1000 # print update every 1000 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'),),
  ) -> scale_pars

df_temp <- data.frame(Task = c('Gesture Recognition'),  
                      N = c(for_stan$N_ids),
                      p_rope_session = c(as.numeric(rope(scale_pars$beta_session,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_session = c(quantile(scale_pars$beta_session, probs=0.025)), 
                      upper_CI_session = c(quantile(scale_pars$beta_session, probs=0.975)),
                      p_rope_inter1 = c(as.numeric(rope(scale_pars$beta_diff_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.025)), 
                      upper_CI_inter1 = c(quantile(scale_pars$beta_diff_inter, probs=0.975)),
                      p_rope_inter2 = c(as.numeric(rope(scale_pars$beta_hours_inter,ci=1, range=c(-0.18, 0.18)))),
                      lower_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.025)), 
                      upper_CI_inter2 = c(quantile(scale_pars$beta_hours_inter, probs=0.975))
)

rownames(df_temp) = NULL

df_results <- rbind(df_results,df_temp)

df = df_clean


############################## COMPILE RESULTS ---------------------------------


df_temp_all <- df_results
temp <- df_temp_all

p_rope_session_adjusted <- p.adjust(df_results$p_rope_session, method="BH")
p_rope_inter1_adjusted <- p.adjust(df_results$p_rope_inter1, method="BH")
p_rope_inter2_adjusted <- p.adjust(df_results$p_rope_inter1, method="BH")

df_temp_all <- cbind(df_temp_all, p_rope_session_adjusted,p_rope_inter1_adjusted,p_rope_inter2_adjusted) 

df_temp_all %>%
  mutate(across(where(is.numeric), \(x) round(x,2))) %>%
  mutate(p_rope_session_adjusted = replace(p_rope_session_adjusted, p_rope_session_adjusted == 0, '<.001')) %>%
  mutate(p_rope_inter1_adjusted = replace(p_rope_inter1_adjusted, p_rope_inter1_adjusted == 0, '<.001')) %>%
  mutate(p_rope_inter2_adjusted = replace(p_rope_inter2_adjusted, p_rope_inter2_adjusted == 0, '<.001')) -> df_temp_all

write.csv(df_temp_all, here::here(paste(models_folder,"diurnal_beta_coeffs.csv",sep='/')))


