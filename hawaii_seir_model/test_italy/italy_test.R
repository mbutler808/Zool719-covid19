library(tidyverse)
library(pomp)
library(ggplot2)

data = read.csv("time_series_covid19_confirmed_global.csv")
data = data %>% 
  select(-c(Province.State, Lat, Long)) %>%
  filter(Country.Region == "Italy") %>%
  pivot_longer(cols = -1, names_to = "date", values_to = "C") %>%
  transmute(date = as.Date(str_remove_all(date, "X"), format = "%m.%d.%y"),
            C, day = c(1:433))

data$C <- with(data, c(data$C[1], data$C[-1]-data$C[-nrow(data)]))


ggplot(data, aes(x = date, y = C)) + geom_line() + 
  ylab("New Cases") + ggtitle("Daily Confirmed Cases of COVID-19 in Italy")

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
  simulate(params = c(Beta = 10, mu_EI = 0.001, mu_IR = .02, k = 0.42,
                      rho = 400, eta = 0.4, N = 30000),
           nsim = 20, format = "data.frame", include = TRUE)

ggplot(sims, aes(x = day, y = C, group = .id, color = .id=="data")) +
  geom_line() + guides(color=FALSE)