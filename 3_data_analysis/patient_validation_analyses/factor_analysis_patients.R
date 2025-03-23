# ============================================================================ #
#  Script: Factor Analysis and Data Exploration
#                                                                      
#  Updated on: 5th of April                                  
#  Author: Dragos Gruia                                
#                                                                      
#  Description:                                                           
#    This script performs data preparation, exploratory factor analysis,
#    and correlation analyses on cognitive and demographic data from a  
#    normative cohort. The workflow includes:                         
#      - Loading and formatting data from multiple CSV files            
#      - Creating a combined dataset for cognitive subtests and MOCA     
#      - Scaling and winsorizing the data                                
#      - Generating a correlation matrix and visualizing it                      
#      - Assessing the suitability of the data for factor analysis        
#      - Determining the optimal number of factors via parallel analysis  
#      - Running factor analysis using omega and generating factor scores 
#      - Comparing factor scores between healthy controls and patients     
#      - Plotting group differences and correlations between MOCA, IADL,   
#        and global IC3 scores                                           
#                                                                      
#  Inputs:                                                        
#    - ic3_healthy_cleaned_cog_and_demographics.csv                      
#         Normative dataset with cognitive and demographic variables     
#    - IC3_summaryScores_and_MOCA.csv                                    
#         Dataset including MOCA/IADL scores and patient cognitive measures
#                                                                      
#  Outputs:                                                       
#    - Correlation matrix plot of cognitive subtests                     
#    - Factor analysis output (omega analysis, factor scores)            
#    - Group comparison statistics (t-tests, effect size)                
#    - ggplot2 visualizations comparing global IC3 scores with MOCA and IADL
#    - Correlation coefficients (r, p-values, R-squared)                  
#    - Linear model summaries predicting IADL using global IC3 and MOCA     
#                                                                      
# ============================================================================ #

library(tidyr)
library(dplyr)
library(psych)
library(factoextra)
library(corrplot)
library(pcaMethods)
library(corrr)
library(datawizard)
library(stats)
library(stringr)
library(magrittr)

options(mc.cores = 8)

############################ LOAD AND FORMAT DATA ------------------------------

df = read.csv('ic3_healthy_cleaned_cog_and_demographics.csv')
df_all_temp <- subset(df, select=auditoryAttention:spatialSpan)
df_all_temp %<>% select(-trail, -trail2,-trail3)
df_all_temp <- subset(df_all_temp, rowSums(is.na(df_all_temp)) != ncol(df_all_temp))
df_all_temp %<>% select(order(colnames(df_all_temp)))

df_moca = read.csv('IC3_summaryScores_and_MOCA.csv')
df_moca = df_moca[!is.na(df_moca[,'srt']),]
df_moca = df_moca[df_moca$user_id != 'ic3study00089-session1-versionA',] # excluded due to technical errors

df_moca = df_moca %>% drop_na(Moca.total..30.)
users_patients = df_moca$user_id

df_moca = subset(df_moca, select=orientation:trailAll)
df_moca %<>% select(order(colnames(df_moca))) %>% select(-trail1, -trail2,-trail3)

tasks_order = c('orientation','taskRecall','pal','digitSpan','spatialSpan',
                'comprehension', 'semantics', 'blocks', 'trailAll', 'oddOneOut',
                'ided', 'pear', 'srt', 'auditoryAttention', 'crt', 'motorControl',
                'calculation', 'gesture')
task_names = c('Orientation', 'Task Recall', 'Paired Associates Learning',
               'Digit Span','Spatial Span','Language Comprehension','Semantic Judgement',
               'Blocks','Trail-making','Odd One Out','Rule Learning','Pear Cancellation',
               'Simple Reaction Time','Auditory Attention','Choice Reaction Time',
               'Motor Control','Graded Calculation','Gesture Recognition')

df_all <- rbind(df_all_temp,df_moca)
df_all <- df_all[,tasks_order]
colnames(df_all) =task_names

df_all <- scale(df_all)

df_all[,'Simple Reaction Time'] = df_all[,'Simple Reaction Time']*-1
df_all[,'Trail-making'] = df_all[,'Trail-making']*-1

df_all = winsorize(df_all, threshold = c(-5,5), method='raw')


##################### CREATE A CORRELATION MATRIX ------------------------------

cor_subtests <- cor(df_all, method='pearson',use="pairwise.complete.obs")
cor_subtests <- round(cor_subtests, 2)
corrplot(cor_subtests, is.corr=TRUE)

############ CHECK IF DATA IS APPROPRIATE FOR FACTOR ANALYSIS ------------------

KMO(df_all)

cortest.bartlett(df_all)

################# ASSESS OPTIMAL NUMBER OF FACTORS -----------------------------

fa.parallel(df_all, fa='fa') 

################# RUN FACTOR ANALYSIS ------------------------------------------

b4 = omega(cor_subtests, nfactors = 6, rotate='oblimin', fm='ml', digits=3, n.obs=dim(df_all)[1])
b4

omega.diagram(b4, digits = 2)

factor_scores = factor.scores(df_all, b4$schmid$sl[,1:7], rho=cor_subtests, method='tenBerge')$scores

# Separate factor scores between healthy and patients and check for group differences
g_patients = factor_scores[(dim(df_all_temp)[1]+1):dim(df_all)[1],1]
g_healthy = factor_scores[1:dim(df_all_temp)[1],1]

group = c(rep('Controls',length(g_healthy)),
          rep('Patients',length(g_patients)))

group_comparison = data.frame(group,factor_scores[,1])
colnames(group_comparison) = c('group','g')
group_comparison %<>% drop_na()

t.test(g~group,var.equal=TRUE, data=group_comparison)

effect_d = (mean(g_healthy,na.rm=TRUE) - mean(g_patients,na.rm=TRUE))/sd(group_comparison$g,na.rm=TRUE)
effect_d

# Plot factor scores separately for healthy and patients
ggplot(group_comparison, aes(x=group,y=g)) + 
  geom_violin() +
  theme_classic() 

ggplot(group_comparison, aes(x=group,y=g)) + 
  geom_boxplot() +
  theme_classic() +
  ylab('IC3 global score') + 
  xlab('') + 
  theme(text=element_text(size=25))+
  geom_jitter(shape=16, position=position_jitter(0.2),alpha=0.5)


################### PLOT CORRELATION BETWEEN MOCA AND GLOBAL IC3 scores --------

df_moca['global_ic3'] =g_patients
df_moca['user_id'] = users_patients

df_patients = read.csv('IC3_summaryScores_and_MOCA.csv')
df_patients = left_join(df_moca,subset(df_patients, select=c('Moca.total..30.','IADL','user_id')),
                        by= c("user_id" = "user_id"))

g = ggplot(df_patients, aes(Moca.total..30.,global_ic3)) + 
  geom_point(shape=16, size=3,color='#7851A9') +
  geom_smooth(method = lm, formula = y ~ x, color='#ffd400', fill='#fff192') +
  geom_hline(yintercept = 0.16,linetype="dashed") +
  geom_vline(xintercept = 26,linetype="dashed") +
  ylab('IC3 global score') + # for the x axis label 
  xlab('MOCA') + # for the y axis label
  theme_classic()+
  theme(text=element_text(size=25))


g + annotate(x=26,y=-5,label="MOCA cut-off",vjust=0,geom="label",size=5) +
  annotate(x=15,y=0.16,label="Mean of controls",vjust=-0.5,geom="label",size=5) +
  annotate("text", x = 6.5, y = 1.3, size=7, label = "italic(R) == 0.58", parse=TRUE) +
  annotate("text", x = 6.6, y = 0.75, size=7, label = "italic(P) < 0.001", parse=TRUE) 
#  annotate('rect', xmin=5, xmax=30, ymin=-0.93, ymax=1.09, alpha=.2, fill='#d9d9d9')


r_value = stats::cor(df_patients$Moca.total..30., df_patients$global_ic3, use='pairwise.complete.obs')
r_value
p_value = cor.test(df_patients$Moca.total..30. , df_patients$global_ic3, use='pairwise.complete.obs')$p.value
p_value
r_squared = r_value*r_value
r_squared

################### PLOT CORRELATION BETWEEN IADL AND GLOBAL IC3 scores --------

g = ggplot(df_patients, aes(IADL,global_ic3)) + 
  geom_point(shape=16, size=3,color='#7851A9') +
  geom_hline(yintercept = 0.16,linetype="dashed") +
  geom_smooth(method = lm, formula = y ~ x, color='#ffd400', fill='#fff192') +
  ylab('IC3 global score') + # for the x axis label 
  xlab('Instrumental activities of daily living') + # for the y axis label
  theme_classic() +
  theme(text=element_text(size=25))

g + annotate("text", x = 1.3, y = 1, size=7, label = "italic(R) == 0.51", parse=TRUE) +
  annotate(x=4,y=-0.16,label="Mean of controls",vjust=-0.5,geom="label",size=5) +
  annotate("text", x = 1.35, y = 0.55, size=7, label = "italic(P) < 0.001", parse=TRUE) 

r_value = cor(df_patients$IADL , df_patients$global_ic3, use='pairwise.complete.obs',method='pearson')
r_value
p_value = cor.test(df_patients$IADL , df_patients$global_ic3, use='pairwise.complete.obs')$p.value
p_value
r_squared = r_value*r_value
r_squared

r_value = cor(df_patients$IADL , df_patients$Moca.total..30., use='pairwise.complete.obs')
r_value
p_value = cor.test(df_patients$IADL ,  df_patients$Moca.total..30., use='pairwise.complete.obs')$p.value
p_value
r_squared = r_value*r_value
r_squared

############# PREDICT IADL USING MOCA AND IC3_global ---------------------------

library(lme4)

model <- lm(IADL ~ global_ic3, data=df_patients)
summary(model)  

model <- lm(IADL ~ global_ic3 + Moca.total..30., data=df_patients)
summary(model)  

model <- lm(IADL ~ global_ic3 + Moca.total..30. + global_ic3*Moca.total..30., data=df_patients)
summary(model)  
