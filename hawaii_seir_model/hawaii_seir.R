library(tidyverse)
library(pomp)
library(ggplot2)

COUNTY = "Honolulu"

data = read.csv("hawaii_covid_cases.csv")
data = data %>% 
  select(-c(New.Positive.Tests, Total.Test.Encounters)) %>%
  filter(County == COUNTY)
  data <- data[-1]  
 
#Add 'day' to full data set
data$Date <- as.Date(data$Date, format = "%m/%d/%Y")
size <- dim(data)[1]
data$day <- c(1:size)
  
names(data) <- c("date", "C", "day")

data$C <- cumsum(data$C)


####
#Set model values
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




########################################
##TIME 1 - Feb. 15 to first lockdown 
data_t1 <- data[1:22,] #from start until March 22nd

ggplot(data_t1, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))

covid_rinit_t1 = "
S = 2500;
E = 1;
I = 1;
R = 0;
"

covid_t1 <- pomp(data = data_t1, times = "day", t0 = 0,
              rprocess = euler(step.fun = Csnippet(covid_rprocess), delta.t = 1/7),
              rmeasure = Csnippet(covid_rmeasure),
              dmeasure = Csnippet(covid_dmeasure),
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
  simulate(params = c(Beta = 250, mu_EI = .05, mu_IR = 0.005, k = 0.2,
                      rho = .5, eta=0, N = 900000),
           nsim = 20, format = "data.frame", include = TRUE)

sims_t1$date <- c(data$date[1:22], rep(data$date[1:22], each=20))

dat <- sims_t1

ggplot(dat, aes(x = date, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE) + labs(x = "Date") + labs(y = "Cases")


t_s <- round(mean(dat$S, na.rm =T))
t_e <- round(mean(dat$E, na.rm =T))
t_i <- round(mean(dat$I, na.rm =T))
t_r <- round(mean(dat$R, na.rm =T))

#############################
#TIME 1.5
data_t1.5 <- data[23:54,] #First lockdown begins. From March 23 - May 31st

ggplot(data_t1.5, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_rinit_t1.5 = "
S = 2476;
E = 20;
I = 6;
R = 0;
"

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
  simulate(params = c(Beta = 100, mu_EI = 0.006, mu_IR = .0035, k = 0.6,
                      rho = 20, eta = 0, N = 900000),
           nsim = 20, format = "data.frame", include = TRUE)

sims_t1.5$date <- c(data$date[23:54], rep(data$date[23:54], each=20))

dat <- sims_t1.5

ggplot(dat, aes(x = date, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE) + labs(x = "Date") + labs(y = "Cases")

t_s <- round(mean(dat$S, na.rm =T))
t_e <- round(mean(dat$E, na.rm =T))
t_i <- round(mean(dat$I, na.rm =T))
t_r <- round(mean(dat$R, na.rm =T))


#############################
#TIME 1.6
data_t1.6 <- data[55:92,] #First lockdown begins. From March 23 - May 31st

ggplot(data_t1.6, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_rinit_t1.6 = "
S = 2360;
E = 121;
I = 20;
R = 0;
"

covid_t1.6 <- pomp(data = data_t1.6, times = "day", t0 = 0,
                   rprocess = euler(step.fun = Csnippet(covid_rprocess), delta.t = 1/7),
                   rmeasure = Csnippet(covid_rmeasure),
                   dmeasure = Csnippet(covid_dmeasure),
                   partrans = parameter_trans( 
                     log=c("Beta","mu_EI","mu_IR", "k", "rho")),
                   obsnames = covid_obsnames,
                   statenames = covid_statenames,
                   paramnames = covid_paramnames,
                   rinit = Csnippet(covid_rinit_t1.6)
)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

sims_t1.6 = covid_t1.6 %>%
  simulate(params = c(Beta = 10, mu_EI = 0.0005, mu_IR = .00035, k = 0.42,
                      rho = 12, eta = 0, N = 900000),
           nsim = 20, format = "data.frame", include = TRUE)

sims_t1.6$date <- c(data$date[55:92], rep(data$date[55:92], each=20))

dat <- sims_t1.6

ggplot(dat, aes(x = date, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE) + labs(x = "Date") + labs(y = "Cases")


t_s <- round(mean(dat$S, na.rm =T))
t_e <- round(mean(dat$E, na.rm =T))
t_i <- round(mean(dat$I, na.rm =T))
t_r <- round(mean(dat$R, na.rm =T))

########################
#set t2
#June 1, 2020 to July 31, 2020
#end of stay at home order, beaches/restaurants open
#close again

data_t2 <- data[93:153,]

ggplot(data_t2, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_rinit_t2 = "
S = 2352;
E = 125;
I = 24;
R = 1;
"

### There is no covid_rprocess_t2, _t3, etc. so I replaced them all with covid_rprocess_t1

covid_t2 <- pomp(data = data_t2, times = "day", t0 = 0,
              rprocess = euler(step.fun = Csnippet(covid_rprocess), delta.t = 1/7),
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
  simulate(params = c(Beta = 30, mu_EI = 0.1, mu_IR = .0004, k = 0.07,
                      rho = .03, eta = 0, N = 9000000),
           nsim = 20, format = "data.frame", include = TRUE)

sims_t2$date <- c(data$date[93:153], rep(data$date[93:153], each=20))

dat <- sims_t2

ggplot(dat, aes(x = date, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE) + labs(x = "Date") + labs(y = "Cases")

t_s <- round(mean(dat$S, na.rm =T))
t_e <- round(mean(dat$E, na.rm =T))
t_i <- round(mean(dat$I, na.rm =T))
t_r <- round(mean(dat$R, na.rm =T))



########################
#set t3
#August 1, 2020 to October 15, 2020
#Limited social gatherings
#second stay at home order on August 27
#Safe travels start October 15
 
data_t3 <- data[154:229,]

ggplot(data_t3, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_rinit_t3 = "
S = 3500;
E = 200;
I = 150;
R = 100;
"

covid_t3 <- pomp(data = data_t3, times = "day", t0 = 0,
              rprocess = euler(step.fun = Csnippet(covid_rprocess), delta.t = 1/7),
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
  simulate(params = c(Beta = 3, mu_EI = 0.009, mu_IR = .04, k = 0.42,
                      rho = 20, eta = 0.3, N = 900000),
           nsim = 20, format = "data.frame", include = TRUE)

ggplot(sims_t3, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)

t3_s <- round(mean(sims_t3$S, na.rm =T))
t3_e <- round(mean(sims_t3$E, na.rm =T))
t3_i <- round(mean(sims_t3$I, na.rm =T))
t3_r <- round(mean(sims_t3$R, na.rm =T))






########################
#set t4
#October 16, 2020 to December 15, 2020
#Start of Safe travels program to first administered vaccine
 
data_t4 <- data[230:290,]

ggplot(data_t4, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_rinit_t4 = "
S = 2000;
E = 250;
I = 200;
R = 200;
"

covid_t4 <- pomp(data = data_t4, times = "day", t0 = 0,
              rprocess = euler(step.fun = Csnippet(covid_rprocess), delta.t = 1/7),
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


t4_s <- round(mean(sims_t4$S, na.rm =T))
t4_e <- round(mean(sims_t4$E, na.rm =T))
t4_i <- round(mean(sims_t4$I, na.rm =T))
t4_r <- round(mean(sims_t4$R, na.rm =T))




########################
#set t5
#December 16, 2020 to March 28, 2021
#First vaccine administration to present
 
data_t5 <- data[291:393,]

ggplot(data_t5, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_rinit_t5 = "
S = 3000;
E = 200;
I = 200;
R = 20000;
"

covid_t5 <- pomp(data = data_t5, times = "day", t0 = 0,
              rprocess = euler(step.fun = Csnippet(covid_rprocess), delta.t = 1/7),
              rmeasure = Csnippet(covid_rmeasure),
              dmeasure = Csnippet(covid_dmeasure),
              partrans = parameter_trans( 
                log=c("Beta","mu_EI","mu_IR", "k", "rho")),
              obsnames = covid_obsnames,
              statenames = covid_statenames,
              paramnames = covid_paramnames,
              rinit = Csnippet(covid_rinit_t5)
)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

sims_t5 = covid_t5 %>%
  simulate(params = c(Beta = 1.5, mu_EI = 0.008, mu_IR = .045, k = 0.42,
                      rho = 125, eta = 0.3, N = 15000),
           nsim = 20, format = "data.frame", include = TRUE)

ggplot(sims_t5, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)

t5_s <- round(mean(sims_t5$S, na.rm =T))
t5_e <- round(mean(sims_t5$E, na.rm =T))
t5_i <- round(mean(sims_t5$I, na.rm =T))
t5_r <- round(mean(sims_t5$R, na.rm =T))






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


#Merge plots into a single image
#install.packages("cowplot")
library(cowplot)
#install.packages("ggpubr")
library(ggpubr)

a <- ggplot(sims_t5, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)
b <- ggplot(sims_t1.5, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)
c <- ggplot(sims_t5, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)
d <- ggplot(sims_t5, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)
e <- ggplot(sims_t5, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)
f <- ggplot(sims_t5, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)

ggarrange(a,b,c,d,e,f)
ggarrange(a,b,c,d,e,f,
          labels = c("A", "B", "C", "D", "E", "F"),
          ncol = 3, nrow =2)

#Relabel each graph as "1", "2", etc. 
ggarrange(a,b,c,d,e,f,
          labels = c("1", "2", "3", "4", "5", "6"),
          ncol = 3, nrow =2)




###########################Vaccine data
#input vaccine data
vax <- read.csv("Hawaii_vaccine_data.csv", fileEncoding = "UTF-8-BOM")

#merge vax data with cases dataframe (% population partially and fully vaxxed)
vax1 <- head(vax, -12)
data_t5 <- cbind(data_t5, new_col = vax1$percent_partial)
names(data_t5)[names(data_t5) == "new_col"] <- "percent_partial_vax"
data_t5 <- cbind(data_t5, new_col = vax1$percent_full)
names(data_t5)[names(data_t5) == "new_col"] <- "percent_full_vax"

#names(data_t5) <- c("date", "C", "day", "%pvax", "%fvax)
