library(tidyverse)
library(pomp)
library(ggplot2)


data = read.csv("hawaii_covid_cases.csv")
data = data %>% 
  select(-c(New.Positive.Tests, Total.Test.Encounters)) %>%
  filter(County == "Honolulu")
data <- data[-1]  

data$Date <- as.Date(data$Date, format = "%m/%d/%Y")

data$day <- c(1:393)

names(data) <- c("date", "C", "day")


# CLOSED POPULATION EXAMPLE
closed.sir.ode <- Csnippet("
  DS = -Beta*S*I/N;
  DI = Beta*S*I/N-gamma*I;
  DR = gamma*I;
")

init1 <- Csnippet("
  S = N-1;
  I = 1;
  R = 0;
  ")

pomp(data=data.frame(time= data$day, data=data$C),
     times = "time", t0=0,
     skeleton=vectorfield(closed.sir.ode),
     rinit=init1,
     statenames=c("S","I","R"),
     paramnames=c("Beta","gamma","N")) -> closed.sir

params1 <- c(Beta=7,gamma=1/13,N=1)

x <- trajectory(closed.sir,params=params1,format="data.frame")

ggplot(data=x,mapping=aes(x=time,y=I))+geom_line()



























# OPEN POPULATION EXAMPLE
open.sir.ode <- Csnippet("
  DS = -Beta*S*I/N+mu*(N-S);
  DI = Beta*S*I/N-gamma*I-mu*I;
  DR = gamma*I-mu*R;
")

init2 <- Csnippet("
  S = S_0;
  I = I_0;
  R = N-S_0-I_0;
")

pomp(data=data.frame(time=seq(0,20,by=1/52),cases=NA),
     times="time",t0=-1/52,
     skeleton=vectorfield(open.sir.ode),
     rinit=init2,
     statenames=c("S","I","R"),
     paramnames=c("Beta","gamma","mu","S_0","I_0","N")
) -> open.sir

params3 <- c(mu=1/50,Beta=400,gamma=365/13,
             N=100000,S_0=100000/12,I_0=100)

x <- trajectory(open.sir,params=params3,format="d")

library(ggplot2)
ggplot(data=x,mapping=aes(x=time,y=I))+geom_line()
ggplot(data=x,mapping=aes(x=S,y=I))+geom_path()