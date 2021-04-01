library(tidyverse)
library(pomp)
library(ggplot2)

COUNTY = "Honolulu"

data = read.csv("hawaii_covid_cases.csv")
data = data %>% 
  select(-c(New.Positive.Tests, Total.Test.Encounters)) %>%
  filter(County == COUNTY)
  data <- data[-1]  

data$Date <- as.Date(data$Date, format = "%m/%d/%Y")

size <- dim(data)[1]

data$day <- c(1:size)

names(data) <- c("date", "C", "day")

data$C <- cumsum(data$C)

ggplot(data, aes(x = date, y = C)) + geom_line() + 
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

covid_rinit = "
S = 2500;
E = 1;
I = 1;
R = 0;
"

covid <- pomp(data = data, times = "day", t0 = 0,
              rprocess = euler(step.fun = Csnippet(covid_rprocess), delta.t = 1/7),
              rmeasure = Csnippet(covid_rmeasure),
              dmeasure = Csnippet(covid_dmeasure),
              partrans = parameter_trans( 
                log=c("Beta","mu_EI","mu_IR", "k", "rho")),
              obsnames = covid_obsnames,
              statenames = covid_statenames,
              paramnames = covid_paramnames,
              rinit = Csnippet(covid_rinit)
)

sims = covid %>%
  simulate(params = c(Beta = 7.75, mu_EI = 0.001, mu_IR = .04, k = 0.42,
                      rho = 400, eta = 0.3, N = 15000),
           nsim = 20, format = "data.frame", include = TRUE)

ggplot(sims, aes(x = day, y = C, group = .id, color = .id=="data")) +
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
