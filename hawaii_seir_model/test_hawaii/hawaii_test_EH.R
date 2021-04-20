require(tidyverse)
require(pomp)
require(ggplot2)

#input vaccine data
vax <- read.csv("Hawaii_vaccine_data.csv", fileEncoding = "UTF-8-BOM")

COUNTY = "Honolulu"

data = read.csv("../hawaii_covid_cases.csv")
data = data %>% 
  select(-c(New.Positive.Tests, Total.Test.Encounters)) %>%
  filter(County == COUNTY)
data <- data[-1]  

data_t1 <- data[1:23,] #from start until March 23
#start to end of first lockdown

data_t1$Date <- as.Date(data_t1$Date, format = "%m/%d/%Y")


size <- dim(data_t1)[1]

data_t1$day <- c(1:size)

names(data_t1) <- c("date", "C", "day")

data_t1$C <- cumsum(data_t1$C)

ggplot(data_t1, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


###Vax plot not showing line :/
# ggplot(vax, aes(full_vax, percent_full)) + geom_line() 
# ggplot(vax, aes(x=date, y=percent_full)) + geom_line() + theme(axis.text.x = element_text(angle = 45, hjust = 1))


covid_statenames = c("S", "E", "I", "R")
covid_paramnames = c("Beta", "mu_EI", "rho", "mu_IR", "N", "eta", "k")
covid_obsnames = "C"

covid_dmeasure_t1 = "lik = dpois(C, rho*I + 1e-6, give_log);"

covid_rmeasure_t1 = "C = rnbinom(rho*I, k);"

covid_rprocess_t1 = "
double dN_SE = rbinom(S, 1-exp(-Beta*I/N*dt));
double dN_EI = rbinom(E, 1-exp(-mu_EI*dt));
double dN_IR = rbinom(I, 1-exp(-mu_IR*dt));
S -= dN_SE;
E += dN_SE - dN_EI;
I += dN_EI - dN_IR;
R += dN_IR;
"
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

covid_rinit_t1 = "
S = 2500;
E = 1;
I = 1;
R = 0;
"

covid_t1 <- pomp(data = data_t1, times = "day", t0 = 0,
                 rprocess = euler(step.fun = Csnippet(covid_rprocess_t1), delta.t = 1/7),
                 rmeasure = Csnippet(covid_rmeasure_t1),
                 dmeasure = Csnippet(covid_dmeasure_t1),
                 partrans = parameter_trans( 
                   log=c("Beta","mu_EI","mu_IR", "k", "rho")),
                 obsnames = covid_obsnames,
                 statenames = covid_statenames,
                 paramnames = covid_paramnames,
                 rinit = Csnippet(covid_rinit_t1)
)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

sims_t1 = covid_t1 %>%
  simulate(params = c(Beta = 20, mu_EI = 0.15, mu_IR = .05, k = 0.4,
                      rho = 2, eta = 0.5, N = 150000),
           nsim = 20, format = "data.frame", include = TRUE)

ggplot(sims_t1, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)

sims_t1
ends <- grep("20",sims_t1$day)
sims_t1_end <- sims_t1[ends,]

t1_s <- round(mean(sims_t1_end$S, na.rm = T))
t1_e <- round(mean(sims_t1_end$E, na.rm = T))
t1_i <- round(mean(sims_t1_end$I, na.rm = T))
t1_r <- round(mean(sims_t1_end$R, na.rm = T))


########################
#set t1.5
# March 23, 2020 to April 23, 2020


data_t1.5 <- data[23:54,] #from start until May 31
#start to end of first lockdown

data_t1.5$Date <- as.Date(data_t1.5$Date, format = "%m/%d/%Y")


size <- dim(data_t1.5)[1]

data_t1.5$day <- c(1:size)

names(data_t1.5) <- c("date", "C", "day")

data_t1.5$C <- cumsum(data_t1.5$C)

ggplot(data_t1.5, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_statenames = c("S", "E", "I", "R")
covid_paramnames = c("Beta", "mu_EI", "rho", "mu_IR", "N", "eta", "k")
covid_obsnames = "C"

covid_dmeasure = "lik = dpois(C, rho*I + 1e-6, give_log);"

covid_rmeasure = "C = rnbinom(rho*I, k);"

covid_rprocess = "
double dN_SE = rbinom(S, 1-exp(-Beta*I/N*dt));
double dN_EI = rbinom(E, 1-exp(-mu_EI*dt));
double dN_IR = rbinom(I, 1-exp(-mu_IR*dt));
S -= dN_SE;
E += dN_SE - dN_EI;
I += dN_EI - dN_IR;
R += dN_IR;
"
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

covid_rinit_t1.5 = "
S = 2478;
E = 11;
I = 10;
R = 4;
"

# covid_rinit_t1.5 = "
# int t1_s;
# S = &t1_s;
# E = 11;
# I = 10;
# R = 4;
# "

covid_t1.5 <- pomp(data = data_t1.5, times = "day", t0 = 0,
                 rprocess = euler(step.fun = Csnippet(covid_rprocess), delta.t = 1/7),
                 rmeasure = Csnippet(covid_rmeasure),
                 dmeasure = Csnippet(covid_dmeasure),
                 partrans = parameter_trans( 
                   log=c("Beta","mu_EI","mu_IR", "k", "rho")),
                 obsnames = covid_obsnames,
                 statenames = covid_statenames,
                 paramnames = covid_paramnames,
                 rinit = Csnippet(covid_rinit_t1.5)
)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

sims_t1.5 = covid_t1.5 %>%
  simulate(params = c(Beta = 15, mu_EI = 0.03, mu_IR = .2, k = 0.42,
                      rho = 1.2, eta = 0.3, N = 15000),
           nsim = 20, format = "data.frame", include = TRUE)

ggplot(sims_t1.5, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)


sims_t1.5
ends <- grep("20",sims_t1.5$day)
sims_t1.5_end <- sims_t1.5[ends,]

t_s <- round(mean(sims_t1.5_end$S, na.rm = T))
t_e <- round(mean(sims_t1.5_end$E, na.rm = T))
t_i <- round(mean(sims_t1.5_end$I, na.rm = T))
t_r <- round(mean(sims_t1.5_end$R, na.rm = T))


########################
#set t2
#June 1, 2020 to July 31, 2020
#end of stay at home order, beaches/restaurants open
#close again

data_t2 <- data[93:153,]

data_t2$Date <- as.Date(data_t2$Date, format = "%m/%d/%Y")


size <- dim(data_t2)[1]

data_t2$day <- c(1:size)

names(data_t2) <- c("date", "C", "day")

#create cummulative case count for entire data set to include cases from previous time point
data_sum <- data

data_sum$Date <- as.Date(data_sum$Date, format = "%m/%d/%Y")

size <- dim(data_sum)[1]

data_sum$day <- c(1:size)

names(data_sum) <- c("date", "C", "day")

data_sum$C <- cumsum(data_sum$C)

data_t2$C[1] <- sum(data_sum$C[92:93])

data_t2$C <- cumsum(data_t2$C)

ggplot(data_t2, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_rinit_t2 = "
S = 1334;
E = 945;
I = 94;
R = 130;
"

### There is no covid_rprocess_t2, _t3, etc. so I replaced them all with covid_rprocess_t1

covid_t2 <- pomp(data = data_t2, times = "day", t0 = 0,
                 rprocess = euler(step.fun = Csnippet(covid_rprocess_t1), delta.t = 1/7),
                 rmeasure = Csnippet(covid_rmeasure),
                 dmeasure = Csnippet(covid_dmeasure),
                 partrans = parameter_trans( 
                   log=c("Beta","mu_EI","mu_IR", "k", "rho")),
                 obsnames = covid_obsnames,
                 statenames = covid_statenames,
                 paramnames = covid_paramnames,
                 rinit = Csnippet(covid_rinit_t2)
)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

sims_t2 = covid_t2 %>%
  simulate(params = c(Beta = 1, mu_EI = 0.01, mu_IR = .04, k = 0.42,
                      rho = 6, eta = 0.3, N = 15000),
           nsim = 20, format = "data.frame", include = TRUE)

ggplot(sims_t2, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)

sims_t2
ends <- grep("20",sims_t2$day)
sims_t2_end <- sims_t2[ends,]

t_s <- round(mean(sims_t2_end$S, na.rm = T))
t_e <- round(mean(sims_t2_end$E, na.rm = T))
t_i <- round(mean(sims_t2_end$I, na.rm = T))
t_r <- round(mean(sims_t2_end$R, na.rm = T))

########################
#set t3
#August 1, 2020 to October 15, 2020
#Limited social gatherings
#second stay at home order on August 27
#Safe travels start October 15

data_t3 <- data[154:229,]

data_t3$Date <- as.Date(data_t3$Date, format = "%m/%d/%Y")


size <- dim(data_t3)[1]

data_t3$day <- c(1:size)

names(data_t3) <- c("date", "C", "day")

data_t3$C[1] <- sum(data_sum$C[153:154])

data_t3$C <- cumsum(data_t3$C)

ggplot(data_t3, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_rinit_t3 = "
S = 1115;
E = 978;
I = 169;
R = 241;
"

covid_t3 <- pomp(data = data_t3, times = "day", t0 = 0,
                 rprocess = euler(step.fun = Csnippet(covid_rprocess_t1), delta.t = 1/7),
                 rmeasure = Csnippet(covid_rmeasure),
                 dmeasure = Csnippet(covid_dmeasure),
                 partrans = parameter_trans( 
                   log=c("Beta","mu_EI","mu_IR", "k", "rho")),
                 obsnames = covid_obsnames,
                 statenames = covid_statenames,
                 paramnames = covid_paramnames,
                 rinit = Csnippet(covid_rinit_t3)
)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

sims_t3 = covid_t3 %>%
  simulate(params = c(Beta = 7, mu_EI = 0.01, mu_IR = .01, k = 0.42,
                      rho = 13, eta = 0.3, N = 15000),
           nsim = 20, format = "data.frame", include = TRUE)

ggplot(sims_t3, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)

sims_t3
ends <- grep("20",sims_t3$day)
sims_t3_end <- sims_t3[ends,]

t_s <- round(mean(sims_t3_end$S, na.rm = T))
t_e <- round(mean(sims_t3_end$E, na.rm = T))
t_i <- round(mean(sims_t3_end$I, na.rm = T))
t_r <- round(mean(sims_t3_end$R, na.rm = T))

########################
#set t4
#October 16, 2020 to December 15, 2020
#Start of Safe travels program to first administered vaccine

data_t4 <- data[230:290,]

data_t4$Date <- as.Date(data_t4$Date, format = "%m/%d/%Y")


size <- dim(data_t4)[1]

data_t4$day <- c(1:size)

names(data_t4) <- c("date", "C", "day")

data_t4$C[1] <- sum(data_sum$C[153:154])

data_t4$C <- cumsum(data_t4$C)

ggplot(data_t4, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_rinit_t4 = "
S = 2000;
E = 250;
I = 200;
R = 200;
"

covid_t4 <- pomp(data = data_t4, times = "day", t0 = 0,
                 rprocess = euler(step.fun = Csnippet(covid_rprocess_t1), delta.t = 1/7),
                 rmeasure = Csnippet(covid_rmeasure),
                 dmeasure = Csnippet(covid_dmeasure),
                 partrans = parameter_trans( 
                   log=c("Beta","mu_EI","mu_IR", "k", "rho")),
                 obsnames = covid_obsnames,
                 statenames = covid_statenames,
                 paramnames = covid_paramnames,
                 rinit = Csnippet(covid_rinit_t4)
)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process

sims_t4 = covid_t4 %>%
  simulate(params = c(Beta = 4.5, mu_EI = 0.01, mu_IR = .04, k = 0.42,
                      rho = 15, eta = 0.3, N = 15000),
           nsim = 20, format = "data.frame", include = TRUE)

ggplot(sims_t4, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)


sims_t4
ends <- grep("20",sims_t4$day)
sims_t4_end <- sims_t4[ends,]

t_s <- round(mean(sims_t4_end$S, na.rm = T))
t_e <- round(mean(sims_t4_end$E, na.rm = T))
t_i <- round(mean(sims_t4_end$I, na.rm = T))
t_r <- round(mean(sims_t4_end$R, na.rm = T))
########################
#set t5
#December 16, 2020 to March 28, 2021
#First vaccine administration to present
####Added vaccine data here
data_t5 <- data[291:393,]

data_t5$Date <- as.Date(data_t5$Date, format = "%m/%d/%Y")

#shorten vax data to match dataframe size
# vax1 <- head(vax, -12)

size <- dim(data_t5)[1]

data_t5$day <- c(1:size)

#merge vax dataframe with cases dataframe
# data_t5 <- cbind(data_t5, new_col = vax1$percent_partial)


names(data_t5) <- c("date", "C", "day")

data_t5$C[1] <- sum(data_sum$C[290:291])

data_t5$C <- cumsum(data_t5$C)

ggplot(data_t5, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_rinit_t5 = "
S = 3000;
E = 200;
I = 200;
R = 20000;
"

covid_t5 <- pomp(data = data_t5, times = "day", t0 = 0,
                 rprocess = euler(step.fun = Csnippet(covid_rprocess_t1), delta.t = 1/7),
                 rmeasure = Csnippet(covid_rmeasure_t1),
                 dmeasure = Csnippet(covid_dmeasure_t1),
                 partrans = parameter_trans( 
                   log=c("Beta","mu_EI","mu_IR", "k", "rho")),
                 obsnames = covid_obsnames,
                 statenames = covid_statenames,
                 paramnames = covid_paramnames,
                 rinit = Csnippet(covid_rinit_t5)
)
spy(covid_t5)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

sims_t5 = covid_t5 %>%
  simulate(params = c(Beta = 1, mu_EI = 0.0001, mu_IR = .0025, k = 0.3,
                      rho = 75, eta = 0.3, N = 15000, dvax=0),
           nsim = 20, format = "data.frame", include = TRUE)

ggplot(sims_t5, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)








pf <- replicate(n=20,logLik(pfilter(covid, Np = 500, 
                                    params = c(Beta = 7.75, mu_EI = 0.001, mu_IR = .04, k = 0.42,
                                               rho = 400, eta = 0.2, N = 15000),
                                    partrans = parameter_trans( 
                                      log = c("Beta", "mu_EI", "mu_IR", "k", "rho")),
                                    dmeasure = Csnippet(covid_dmeasure), 
                                    statenames = covid_statenames,
                                    paramnames = covid_paramnames)))

beta7.75 <- logmeanexp(pf, se =T)

pf <- replicate(n=20,logLik(pfilter(covid, Np = 500, 
                                    params = c(Beta = 8, mu_EI = 0.001, mu_IR = .04, k = 0.42,
                                               rho = 400, eta = 0.2, N = 15000),
                                    partrans = parameter_trans( 
                                      log = c("Beta", "mu_EI", "mu_IR", "k", "rho")),
                                    dmeasure = Csnippet(covid_dmeasure), 
                                    statenames = covid_statenames,
                                    paramnames = covid_paramnames)))

beta8 <- logmeanexp(pf, se =T)
