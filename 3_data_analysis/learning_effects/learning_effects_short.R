
# Updated on 5th of April 2024
# @author: Dragos Gruia

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
library(ggplot2)
library(rstan)
library(posterior)
library(rstanarm)
library(bayestestR)
library(see)
library(lme4)

# Settings
here::here() 
options(mc.cores = 8) 
models_folder = 'models_learning'
output_folder = 'models_learning/fit_save'
cog_data = 'IC3_summaryScores_MT_forBayes.csv'
speech_data = 'data_summaryScore_speech_MT.csv'


# Load data 
df = read.csv(cog_data)
df_clean = df
options(mc.cores = 8) 


################################# 
################################# ANALYSE DATA FOR ORIENTATION
################################# 

df$summary_score = df$orientation 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
    )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_results <- data.frame(Task = c('Orientation'), 
                         N = c(for_stan$N_ids),
                         mean_S1 = mean_s1,
                         mean_S2 = mean_s2,
                         mean_S3 = mean_s3,
                         mean_S4 = mean_s4,
                         beta_standardised = c(mean(scale_pars$time_dif14)), 
                         p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                         lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                         upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                         p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)
rownames(df_results) = NULL
print(df_results)

df = df_clean


################################# 
################################# ANALYSE DATA FOR TASK RECALL
################################# 

df$summary_score = df$taskRecall
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))

  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")


model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  #  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Task Recall'), 
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR PAL
################################# 

to_remove <- (df$ID != 30020) #Get rid of those who failed practice trials
df <- df[to_remove,]

df$summary_score = df$pal
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID)),
  )

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = df$summary_score,
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID
  
)

stan_file = here::here("models_learning_effects/linMixed_learning_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  #  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('PAL'), 
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1)))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean



################################# 
#################################  ANALYSE DATA FOR DIGITS SPAN 
################################# 


df$summary_score = df$digitSpan
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )


df %>%
  ggplot(aes(y = summary_score, x = session, group = ID)) +
  geom_point(color = "cadetblue4", alpha = 0.80) +
  geom_smooth(method = 'lm', se = FALSE, color = "black") +
  facet_wrap(~ID)


for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = df$summary_score,
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID
  
)

stan_file = here::here("models_learning_effects/linMixed_learning_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  #  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Digits Span'), 
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1)))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean

################################# 
#################################  ANALYSE DATA FOR SPATIAL SPAN 
################################# 


df$summary_score = df$spatialSpan 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = df$summary_score,
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID
  
)

stan_file = here::here("models_learning_effects/linMixed_learning_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  #  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Spatial Span'),
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1)))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


################################# 
################################# ANALYSE DATA FOR COMPREHENSION
################################# 


df$summary_score = df$comprehension 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  #  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Comprehension'),
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


################################# 
################################# ANALYSE DATA FOR SEMANTIC JUDGEMENT
################################# 

df$summary_score = df$semantic 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID)),
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  #  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Semantic Judgement'), 
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


################################# 
################################# ANALYSE DATA FOR SPEECH
################################# 


df = read.csv(speech_data)
df_clean = df
options(mc.cores = 8) 

################################# 
#################################  ANALYSE DATA FOR NAMING
################################# 


df$summary_score = df$naming 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1]/df$n_events_naming)
mean_s2 = mean(df$summary_score[df$session==2]/df$n_events_naming)
mean_s3 = mean(df$summary_score[df$session==3]/df$n_events_naming)
mean_s4 = mean(df$summary_score[df$session==4]/df$n_events_naming)

df %<>% 
  mutate(
    age = (age - mean(age,na.rm=TRUE))/sd(age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

df %>%
  ggplot(aes(y = summary_score, x = session, group = ID)) +
  geom_point(color = "cadetblue4", alpha = 0.80) +
  geom_smooth(method = 'lm', se = FALSE, color = "black") +
  facet_wrap(~ID)


for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(df$n_events_naming)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Naming'),
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean

################################# 
#################################  ANALYSE DATA FOR READING
################################# 


df$summary_score = df$reading
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1]/df$n_events_reading)
mean_s2 = mean(df$summary_score[df$session==2]/df$n_events_reading)
mean_s3 = mean(df$summary_score[df$session==3]/df$n_events_reading)
mean_s4 = mean(df$summary_score[df$session==4]/df$n_events_reading)

df %<>% 
  mutate(
    age = (age - mean(age,na.rm=TRUE))/sd(age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID)),
  )

df %>%
  ggplot(aes(y = summary_score, x = session, group = ID)) +
  geom_point(color = "cadetblue4", alpha = 0.80) +
  geom_smooth(method = 'lm', se = FALSE, color = "black") +
  facet_wrap(~ID)


for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(df$n_events_reading)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Reading'), 
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean

################################# 
#################################  ANALYSE DATA FOR REPETITION
################################# 


df$summary_score = df$repetition 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1]/df$n_events_repetition)
mean_s2 = mean(df$summary_score[df$session==2]/df$n_events_repetition)
mean_s3 = mean(df$summary_score[df$session==3]/df$n_events_repetition)
mean_s4 = mean(df$summary_score[df$session==4]/df$n_events_repetition)


df %<>% 
  mutate(
    age = (age - mean(age,na.rm=TRUE))/sd(age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID)),
  )

df %>%
  ggplot(aes(y = summary_score, x = session, group = ID)) +
  geom_point(color = "cadetblue4", alpha = 0.80) +
  geom_smooth(method = 'lm', se = FALSE, color = "black") +
  facet_wrap(~ID)


for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$age,
  sex = df$gender,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(df$n_events_repetition)
  
)

## Run Bayesian Regression 
stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Repetition'), 
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


#########

# SWITCH TO REMAINING COG TASKS

#########

# Load data
df = read.csv(cog_data)
df_clean = df

################################# 
#################################  ANALYSE DATA FOR BLOCKS
################################# 

to_remove <- (df$ID != 30065 & df$ID != 30066 & df$ID != 30077 & df$ID != 30078 & df$ID != 30085) #Get rid of those who failed practice trials
df <- df[to_remove,]

df$summary_score = df$blocks 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = df$summary_score,
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID
  
)

stan_file = here::here("models_learning_effects/linMixed_learning_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  #  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Blocks'),
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1)))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


################################# 
################################# ANALYSE DATA FOR TRAIL
################################# 

df$summary_score = df$trailAll 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Trail-making'), 
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR ODD ONE OUT
################################# 


df$summary_score = df$oddOneOut
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = df$summary_score,
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID
  
)

stan_file = here::here("models_learning_effects/linMixed_learning_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  #  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Odd One out'), 
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1)))
)

scale_pars$time_dif12 = scale_pars$`beta_timepoint[2]` - scale_pars$`beta_timepoint[1]`
df_temp12 <- data.frame(Task = c('Odd One Out'), 
                        N = c(for_stan$N_ids),
                        mean_S1 = mean_s1,
                        mean_S2 = mean_s2,
                        mean_S3 = mean_s3,
                        mean_S4 = mean_s4,
                        beta_standardised = c(mean(scale_pars$time_dif12)), 
                        p_value = sum(scale_pars$time_dif12 <0 )/nrow(scale_pars),
                        lower_CI = c(quantile(scale_pars$time_dif12, probs=0.025)), 
                        upper_CI = c(quantile(scale_pars$time_dif12, probs=0.975)),
                        p_rope = c(as.numeric(rope(scale_pars$time_dif12,ci=1,range = c(-0.18,0.18))))
)

rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


################################# 
################################# ANALYSE DATA FOR IDED
################################# 

to_remove <- (df$ID != 30092) #Get rid of those who failed practice trials
df <- df[to_remove,]

df$summary_score = df$ided 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Rule Learning'), 
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


################################# 
################################# ANALYSE DATA FOR PEAR
################################# 

to_remove <- (df$ID != 30085 & df$ID != 30078) #Get rid of those who failed practice trials
df <- df[to_remove,]

df$summary_score = df$pear 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    summary_score = summary_score*20,
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`

df_temp <- data.frame(Task = c('Pear Cancellation'),
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)

rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR SRT
################################# 

df$summary_score = df$srt 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    summary_score= (summary_score - mean(summary_score,na.rm=TRUE))/sd(summary_score,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = df$summary_score,
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID
  
)

stan_file = here::here("models_learning_effects/linMixed_learning_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  #  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('SRT'), 
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 >0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1)))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean

################################# 
################################# ANALYSE DATA FOR AUDITORY ATTENTION
################################# 

df$summary_score = df$auditoryAttention 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Auditory Attention'),
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


################################# 
#################################  ANALYSE DATA FOR CRT
################################# 

df$summary_score = df$crt
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  #  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

#Prepare summary statistics for output
stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('CRT'),
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 >0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1)))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


################################# 
################################# ANALYSE DATA FOR MOTOR CONTROL
################################# 

df$summary_score = df$motorControl 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Motor Control'), 
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


################################# 
################################# ANALYSE DATA FOR CALCULATION
################################# 


df$summary_score = df$calculation
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Calculation'), 
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


################################# 
################################# ANALYSE DATA FOR GESTURE
################################# 

df$summary_score = df$gesture 
to_remove = df$ID[is.na(df$summary_score) == TRUE]
to_remove = sapply(df$ID, function(x) any(x == to_remove))

df = df[!to_remove, ]

mean_s1 = mean(df$summary_score[df$session==1])
mean_s2 = mean(df$summary_score[df$session==2])
mean_s3 = mean(df$summary_score[df$session==3])
mean_s4 = mean(df$summary_score[df$session==4])

df %<>% 
  mutate(
    Age = (Age - mean(Age,na.rm=TRUE))/sd(Age,na.rm=TRUE),
    session = as.integer(as.factor(session)),
    ID = as.integer(as.factor(ID))
  )

n_events = max(df$summary_score)
n_events <- rep(n_events,times=nrow(df))
df <- cbind(df,n_events)

for_stan = list(
  N = length(df$ID),
  N_ids = length(df$ID)/4,
  N_timepoints = 4,
  summary_score = as.integer(df$summary_score),
  age = df$Age,
  sex = df$Sex,
  timepoint = df$session,
  time = df$time_diff2,
  device_type = df$device_type,
  education_bachelors = df$education_bachelors,
  education_postBachelors = df$education_postBachelors,
  ids = df$ID,
  n_questions = as.integer(n_events)
  
)

## Run Bayesian Regression 

stan_file = here::here("models_learning_effects/binomialMixed_repars.stan")

model_obj = cmdstanr::cmdstan_model( stan_file )

stan_obj_mixed = model_obj$sample(
  data = for_stan,
  seed = 123,
  adapt_delta = 0.95,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 5000,
  refresh = 1000 # print update every 500 iters
)

stan_obj_mixed |>
  as_mcmc.list() %>%
  do.call(rbind, .) |>
  as.data.frame() |>
  select(
    matches(c("beta","sigma",'intercept'))
  ) -> scale_pars


#sum(scale_pars$beta_age >0 )/nrow(scale_pars)
#sum(scale_pars$beta_session <0 )/nrow(scale_pars)
#hist(scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`)

scale_pars$time_dif14 = scale_pars$`beta_timepoint[4]` - scale_pars$`beta_timepoint[1]`
df_temp <- data.frame(Task = c('Gesture Recognition'), 
                      N = c(for_stan$N_ids),
                      mean_S1 = mean_s1,
                      mean_S2 = mean_s2,
                      mean_S3 = mean_s3,
                      mean_S4 = mean_s4,
                      beta_standardised = c(mean(scale_pars$time_dif14)), 
                      p_value = sum(scale_pars$time_dif14 <0 )/nrow(scale_pars),
                      lower_CI = c(quantile(scale_pars$time_dif14, probs=0.025)), 
                      upper_CI = c(quantile(scale_pars$time_dif14, probs=0.975)),
                      p_rope = c(as.numeric(rope(scale_pars$time_dif14,ci=1,range = c(-0.18,0.18))))
)
rownames(df_temp) = NULL
df_results <- rbind(df_results,df_temp)
print(df_results)

df = df_clean


###################
# SAVE RESULTS
###################


df_temp_all <- df_results
p_rope_adjusted <- p.adjust(df_results$p_rope, method="BH")
p_adjusted <- p.adjust(df_results$p_value, method="BH")

df_temp_all <- cbind(df_temp_all, p_rope_adjusted,p_adjusted) 

df_temp_all %>%
  mutate(across(where(is.numeric), \(x) round(x,2))) %>%
  mutate(p_rope_adjusted = replace(p_rope_adjusted, p_rope_adjusted == 0, '<.001')) -> df_temp_all

df_temp_all
write.csv(df_temp_all, "models_learning_effects/beta_coeffs.csv")
