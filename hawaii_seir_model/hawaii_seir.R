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
covid_dmeasure = Csnippet("lik = dpois(C, rho*I + 1e-6, give_log);")
covid_rmeasure = Csnippet("C = rnbinom(rho*I, k);")
covid_rprocess = Csnippet("
double dN_SE = rbinom(S, 1-exp(-Beta*I/N*dt));
double dN_EI = rbinom(E, 1-exp(-mu_EI*dt));
double dN_IR = rbinom(I, 1-exp(-mu_IR*dt));
S -= dN_SE;
E += dN_SE - dN_EI;
I += dN_EI - dN_IR;
R += dN_IR;
")




########################################
##TIME 1 - Feb. 15 to first lockdown 
data_t1 <- data[1:22,] #from start until March 22nd

ggplot(data_t1, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))

covid_rinit_t1 = Csnippet("
S = N-2;
E = 1;
I = 1;
R = 0;
")

covid_t1 <- pomp(data = data_t1, times = "day", t0 = 0,
                 rprocess = euler(step.fun = covid_rprocess, delta.t = 1/7),
                 rmeasure = covid_rmeasure,
                 dmeasure = covid_dmeasure,
                 partrans = parameter_trans( 
                   log=c("Beta","mu_EI","mu_IR", "k", "rho")),
                 obsnames = covid_obsnames,
                 statenames = covid_statenames,
                 paramnames = covid_paramnames,
                 rinit = covid_rinit_t1
)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

sims_t1 = covid_t1 %>%
  simulate(params = c(Beta = 4, mu_EI = .01, mu_IR = 0.02, k = 0.42,
                      rho = 1.2, eta=0, N = 900000),
           nsim = 20, format = "data.frame", include = TRUE)

sims_t1$date <- c(data$date[1:22], rep(data$date[1:22], each=20))

dat <- sims_t1

a <- ggplot(dat, aes(x = date, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE) + labs(x = "Date") + labs(y = "Cases")

a

t_s <- round(mean(dat$S, na.rm =T))
t_e <- round(mean(dat$E, na.rm =T))
t_i <- round(mean(dat$I, na.rm =T))
t_r <- round(mean(dat$R, na.rm =T))

#############################
# Redefines model parameters to incorporate estimates from previous time point
covid_statenames = c("S", "E", "I", "R")
covid_paramnames = c("Beta", "mu_EI", "rho", "mu_IR", "N", "eta", "k", "s", "e", "i", "r")
covid_obsnames = "C"
covid_dmeasure = Csnippet("lik = dpois(C, rho*I + 1e-6, give_log);")
covid_rmeasure = Csnippet("C = rnbinom(rho*I, k);")
covid_rprocess = Csnippet("
double dN_SE = rbinom(S, 1-exp(-Beta*I/N*dt));
double dN_EI = rbinom(E, 1-exp(-mu_EI*dt));
double dN_IR = rbinom(I, 1-exp(-mu_IR*dt));
S -= dN_SE;
E += dN_SE - dN_EI;
I += dN_EI - dN_IR;
R += dN_IR;
")
#############################
# Time 1.5

data_t1.5 <- data[23:54,] #First lockdown begins. From March 23 - May 31st

ggplot(data_t1.5, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_rinit_t = Csnippet("
S = (int) s;
E = (int) e;
I = (int) i;
R = (int) r;
")

covid_t1.5 <- pomp(data = data_t1.5, times = "day", t0 = 0,
                   rprocess = euler(step.fun = covid_rprocess, delta.t = 1/7),
                   rmeasure = covid_rmeasure,
                   dmeasure = covid_dmeasure,
                   partrans = parameter_trans( 
                     log=c("Beta","mu_EI","mu_IR", "k", "rho")),
                   obsnames = covid_obsnames,
                   statenames = covid_statenames,
                   paramnames = covid_paramnames,
                   rinit = covid_rinit_t
)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

sims_t1.5 = covid_t1.5 %>%
  simulate(params = c(Beta = .5, mu_EI = 0.01, mu_IR = .02, k = 0.8,
                      rho = 5, eta = 0, N = 900000, s = t_s, e = t_e, i = t_i, r = t_r),
           nsim = 20, format = "data.frame", include = TRUE)

sims_t1.5$date <- c(data$date[23:54], rep(data$date[23:54], each=20))

dat <- sims_t1.5

b <- ggplot(dat, aes(x = date, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE) + labs(x = "Date") + labs(y = "Cases")

b

t_s <- round(mean(dat$S, na.rm =T))
t_e <- round(mean(dat$E, na.rm =T))
t_i <- round(mean(dat$I, na.rm =T))
t_r <- round(mean(dat$R, na.rm =T))

#############################
#TIME 1.6
data_t1.6 <- data[55:92,] #First lockdown begins. From March 23 - May 31st

ggplot(data_t1.6, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_t1.6 <- pomp(data = data_t1.6, times = "day", t0 = 0,
                   rprocess = euler(step.fun = covid_rprocess, delta.t = 1/7),
                   rmeasure = covid_rmeasure,
                   dmeasure = covid_dmeasure,
                   partrans = parameter_trans( 
                     log=c("Beta","mu_EI","mu_IR", "k", "rho")),
                   obsnames = covid_obsnames,
                   statenames = covid_statenames,
                   paramnames = covid_paramnames,
                   rinit = covid_rinit_t
)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

sims_t1.6 = covid_t1.6 %>%
  simulate(params = c(Beta = .03, mu_EI = 0.005, mu_IR = .02, k = 0.42,
                      rho = 1, eta = 0, N = 900000, s = t_s, e = t_e, i = t_i, r = t_r),
           nsim = 20, format = "data.frame", include = TRUE)

sims_t1.6$date <- c(data$date[55:92], rep(data$date[55:92], each=20))

dat <- sims_t1.6

c <- ggplot(dat, aes(x = date, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE) + labs(x = "Date") + labs(y = "Cases")

c

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
              rinit = Csnippet(covid_rinit_t)
)

#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

sims_t2 = covid_t2 %>%
  simulate(params = c(Beta = 2.8, mu_EI = 0.01, mu_IR = .04, k = 0.42,
                      rho = .07, eta = 0, N = 9000000, s = t_s, e = t_e, i = t_i, r = t_r),
           nsim = 20, format = "data.frame", include = TRUE)

sims_t2$date <- c(data$date[93:153], rep(data$date[93:153], each=20))

dat <- sims_t2

d <- ggplot(dat, aes(x = date, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE) + labs(x = "Date") + labs(y = "Cases")

d

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


covid_t3 <- pomp(data = data_t3, times = "day", t0 = 0,
              rprocess = euler(step.fun = covid_rprocess, delta.t = 1/7),
              rmeasure = covid_rmeasure,
              dmeasure = covid_dmeasure,
              partrans = parameter_trans( 
                log=c("Beta","mu_EI","mu_IR", "k", "rho")),
              obsnames = covid_obsnames,
              statenames = covid_statenames,
              paramnames = covid_paramnames,
              rinit = covid_rinit_t
)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

sims_t3 = covid_t3 %>%
  simulate(params = c(Beta = .1, mu_EI = 0.01, mu_IR = .02, k = 0.42,
                      rho = .065, eta = 0, N = 900000, s = t_s, e = t_e, i = t_i, r = t_r),
           nsim = 20, format = "data.frame", include = TRUE)

sims_t3$date <- c(data$date[154:229], rep(data$date[154:229], each=20))

dat <- sims_t3

e <- ggplot(dat, aes(x = date, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE) + labs(x = "Date") + labs(y = "Cases")

e

t_s <- round(mean(sims_t3$S, na.rm =T))
t_e <- round(mean(sims_t3$E, na.rm =T))
t_i <- round(mean(sims_t3$I, na.rm =T))
t_r <- round(mean(sims_t3$R, na.rm =T))






########################
#set t4
#October 16, 2020 to December 15, 2020
#Start of Safe travels program to first administered vaccine
 
data_t4 <- data[230:290,]

ggplot(data_t4, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_t4 <- pomp(data = data_t4, times = "day", t0 = 0,
              rprocess = euler(step.fun = covid_rprocess, delta.t = 1/7),
              rmeasure = covid_rmeasure,
              dmeasure = covid_dmeasure,
              partrans = parameter_trans( 
                log=c("Beta","mu_EI","mu_IR", "k", "rho")),
              obsnames = covid_obsnames,
              statenames = covid_statenames,
              paramnames = covid_paramnames,
              rinit = covid_rinit_t
)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process

sims_t4 = covid_t4 %>%
  simulate(params = c(Beta = .1, mu_EI = 0.01, mu_IR = .02, k = .42,
                      rho = .32, eta = t_s, N = 900000, s = t_s, e = t_e, i = t_i, r = t_r),
           nsim = 20, format = "data.frame", include = TRUE)

sims_t4$date <- c(data$date[230:290], rep(data$date[230:290], each=20))

dat <- sims_t4

f <- ggplot(dat, aes(x = date, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE) + labs(x = "Date") + labs(y = "Cases")

f

t_s <- round(mean(sims_t4$S, na.rm =T))
t_e <- round(mean(sims_t4$E, na.rm =T))
t_i <- round(mean(sims_t4$I, na.rm =T))
t_r <- round(mean(sims_t4$R, na.rm =T))




########################
#set t5
#December 16, 2020 to March 28, 2021
#First vaccine administration to present
 
data_t5 <- data[291:393,]

ggplot(data_t5, aes(x = date, y = C)) + geom_line() + 
  ylab("Total Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in", paste(COUNTY))


covid_t5 <- pomp(data = data_t5, times = "day", t0 = 0,
              rprocess = euler(step.fun = covid_rprocess, delta.t = 1/7),
              rmeasure = covid_rmeasure,
              dmeasure = covid_dmeasure,
              partrans = parameter_trans( 
                log=c("Beta","mu_EI","mu_IR", "k", "rho")),
              obsnames = covid_obsnames,
              statenames = covid_statenames,
              paramnames = covid_paramnames,
              rinit = covid_rinit_t
)
#Beta = contact rate
#mu_EI = incubation rate
#rho = reporting rate
#mu_IR = recovery/removed rate
#k = overdispersion in the counts process
#eta = number of susceptible (estimated)

sims_t5 = covid_t5 %>%
  simulate(params = c(Beta = .8, mu_EI = 0.001, mu_IR = .045, k = 0.42,
                      rho = 16, eta = t_s, N = 900000, s = t_s, e = t_e, i = t_i, r = t_r),
           nsim = 20, format = "data.frame", include = TRUE)

sims_t5$date <- c(data$date[291:393], rep(data$date[291:393], each=20))

dat <- sims_t5

g <- ggplot(dat, aes(x = date, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE) + labs(x = "Date") + labs(y = "Cases")

g


allsims <- rbind(sims_t1, sims_t1.5, sims_t1.6, sims_t2, sims_t3, sims_t4, sims_t5)

pdf("hawaii_seir_graphs.pdf", height= 8, width=20)

ggplot(allsims, aes(x = date, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE) + labs(x = "Date") + labs(y = "Cases")

dev.off()

# 
# pf <- replicate(n=20,logLik(pfilter(covid, Np = 500, 
#                            params = c(Beta = 7.75, mu_EI = 0.001, mu_IR = .04, k = 0.42,
#                                       rho = 400, eta = 0.2, N = 15000),
#                            partrans = parameter_trans( 
#                              log = c("Beta", "mu_EI", "mu_IR", "k", "rho")),
#                            dmeasure = Csnippet(covid_dmeasure), 
#                            statenames = covid_statenames,
#                            paramnames = covid_paramnames)))
# 
# beta7.75 <- logmeanexp(pf, se =T)
# 
# pf <- replicate(n=20,logLik(pfilter(covid, Np = 500, 
#                                     params = c(Beta = 8, mu_EI = 0.001, mu_IR = .04, k = 0.42,
#                                                rho = 400, eta = 0.2, N = 15000),
#                                     partrans = parameter_trans( 
#                                       log = c("Beta", "mu_EI", "mu_IR", "k", "rho")),
#                                     dmeasure = Csnippet(covid_dmeasure), 
#                                     statenames = covid_statenames,
#                                     paramnames = covid_paramnames)))
# 
# beta8 <- logmeanexp(pf, se =T)


#Merge plots into a single image
#install.packages("cowplot")
library(cowplot)
#install.packages("ggpubr")
library(ggpubr)

pdf("hawaii_seir_graphs.pdf")
ggarrange(a,b,c,d,e,f,g)
ggarrange(a,b,c,d,e,f,g,
          labels = c("A", "B", "C", "D", "E", "F","G"),
          ncol = 7, nrow =1)
dev.off()

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
