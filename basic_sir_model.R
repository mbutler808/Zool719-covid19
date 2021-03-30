# EPI SLIDE CODE TEST
require(deSolve)


# Epi Curve cumulative prevalence
epi.curve <- expression(1/(1+ (exp(-beta*t)*(1-a0)/a0)))
a0 <- .01
beta <- 0.1
t <- seq(0,100,1)
plot(t,eval(epi.curve),type="l",col="blue",
     xlab="Time", ylab="Cumulative Fraction Infected", ylim=c(0,1))

# The number of new cases per unit time
# The epidemic curve is “bell-shaped”, but not completely symmetric
# t → ∞, everyone in the population becomes infected
a <- eval(epi.curve)
b <- diff(a)
plot(1:100,b,type="l",col="blue",
     xlab="Time", ylab="Incident Fraction Infected")

# Basic SIR function
# Beta is effective contact rate
# nu is the removal rate
pars <- c("beta"=0.05,"nu"=0.075)
times <- seq(0,10,0.1)
y0 <- c(100,1,0)

sir <- function(t,y,p) {
  yd1 <- -p["beta"] * y[1]*y[2]
  yd2 <- p["beta"] * y[1]* y[2] - p["nu"]*y[2]
  yd3 <- p["nu"]*y[2]
  list(c(yd1,yd2,yd3),c(N=sum(y)))
}

sir.out <- lsoda(y0,times,sir,pars)

plot(sir.out[,1],sir.out[,2],type="l",col="blue",xlab="Time",
     ylab="Compartment Size")
lines(sir.out[,1],sir.out[,3],col="green")
lines(sir.out[,1],sir.out[,4],col="red")
legend(8,90,c("S","I","R"),col=c("blue","green","red"),lty=c(1,1,1))

# Hypothetical Disease dynamics
lambda.dyn <- function(t,y,p){
  yd1 <- p["mu"] - (p["mu"]+y[2])*y[1]
  yd2 <- (p["mu"] + p["nu"]) * y[2] * (p["R0"]*y[1] - 1)
  list(c(yd1,yd2))
}

pars <- c("R0"=5,"nu"=1.0,"mu"=0.014)
times <- seq(0,100,.1)
y0 <- c(.999,1e-4)
lambda.out <- lsoda(y0,times,lambda.dyn,pars)
plot(lambda.out[,1],lambda.out[,2],type="l",col="blue",
     xlab="Time",ylab="Fraction Susceptible, x(t)")
abline(h=.2,lty=2,col="red")
