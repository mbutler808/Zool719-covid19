require(tidyverse)
require(pomp)      ## Be sure to load pomp *after* the tidyverse, to avoid name conflicts

##
## Simple Model - Ricker model with stochasticity
##                                      N(t+1) = r N(t)  exp(1 - N(t)/K + e(t))
##                                                 where e(t) ~ N(0, sigma)

simulate(t0=0, times=1:20,
  params=c(r=1.2,K=200,sigma=0.1,N_0=50),
  rinit=function (N_0, ...) {
    c(N=N_0)
  },
  rprocess=discrete_time(
    function (N, r, K, sigma, ...) {
      eps <- rnorm(n=1,mean=0,sd=sigma)
      c(N=r*N*exp(1-N/K+eps))
    },
    delta.t=1
  )
) -> sim1

spy(sim1)
str(sim1)

plot(sim1)
as(sim1,"data.frame")

ggplot(data=as.data.frame(sim1),aes(x=time,y=N))+ geom_line()


##
## The Measurement Model
##                        Yt ~ Poisson (bN(t)), where b ~ amount of effort

# Whatʻs changed here?

simulate(t0=0, times=1:20,
  params=c(r=1.2,K=200,sigma=0.1,N_0=50,b=0.05),
  rinit=function (N_0, ...) {
    c(N=N_0)
  },
  rprocess=discrete_time(
    function (N, r, K, sigma, ...) {
      eps <- rnorm(n=1,mean=0,sd=sigma)
      c(N=r*N*exp(1-N/K+eps))
    },
    delta.t=1
  ),
  rmeasure=function (N, b, ...) {
    c(Y=rpois(n=1,lambda=b*N))
  }
) -> sim2

spy(sim2)
as(sim2, "data.frame")

plot(sim2)


ggplot(data=gather(
  as(sim2,"data.frame"),
  variable,value,-time),
  aes(x=time,y=value,color=variable))+
  geom_line()
  

##
## Multiple simulations 
##

simulate(sim2,nsim=20) -> sims      ## 20 simulations from the same starting parameters as sim2

ggplot(data=gather(
  as.data.frame(sims),
  variable,value,Y,N
),
  aes(x=time,y=value,color=variable,
    group=interaction(.id,variable)))+    ## what is .id doing here?
  geom_line()+
  facet_grid(variable~.,scales="free_y")+
  labs(y="",color="")
  
  
##  
## Simulation on a range of parameter values
##

p <- parmat(coef(sim2),3)
p["sigma",] <- c(0.05,0.25,1)
colnames(p) <- LETTERS[1:3]

simulate(sim2,params=p,format="data.frame") -> sims
sims <- gather(sims,variable,value,Y,N)
ggplot(data=sims,aes(x=time,y=value,color=variable,
  group=interaction(.id,variable)))+
  geom_line()+
  scale_y_log10()+
  expand_limits(y=1)+
  facet_grid(variable~.id,scales="free_y")+
  labs(y="",color="")  
  
  
## Is there any difference between N, Y, as stochasticity increases? Why?

##
## Multiple simulations at multiple paramater values, generate density plots
##  

simulate(sim2,params=p,						# how are multiple simulations specified? Any other differences?
  times=seq(0,3),
  nsim=500,format="data.frame") -> sims
ggplot(data=separate(sims,.id,c("parset","rep")),
  aes(x=N,fill=parset,group=parset,color=parset))+
  geom_density(alpha=0.5)+
  # geom_histogram(aes(y=..density..),position="dodge")+
  facet_grid(time~.,labeller=label_both,scales="free_y")+
  lims(x=c(NA,1000))
  

##
## Example with Parus major (bird) population data
##  

parus       ## built-in data from famous dataset

parus %>%
  ggplot(aes(x=year,y=pop))+
  geom_line()+geom_point()+
  expand_limits(y=0)
  
parus %>%
  pomp(
    times="year", t0=1960,
    rinit=function (N_0, ...) {
      c(N=N_0)
    },
    rprocess=discrete_time(
      function (N, r, K, sigma, ...) {
        eps <- rnorm(n=1,mean=0,sd=sigma)
        c(N=r*N*exp(1-N/K+eps))
      },
      delta.t=1
    ),
    rmeasure=function (N, b, ...) {
      c(pop=rpois(n=1,lambda=b*N))
    }
  ) -> rick  
  
  
## what did the above do? What might you want to do with it?


##
## Now try continuous time process
##  

vpstep <- function (N, r, K, sigma, delta.t, ...) {
  dW <- rnorm(n=1,mean=0,sd=sqrt(delta.t))
  c(N = N + r*N*(1-N/K)*delta.t + sigma*N*dW)
}

rick %>% pomp(rprocess=euler(vpstep,delta.t=1/365)) -> vp    # note time step

# can we plot this pomp object?



vp %>%
  simulate(
    params=c(r=0.5,K=2000,sigma=0.1,b=0.1,N_0=2000),
    format="data.frame", include.data=TRUE, nsim=5) %>%
  mutate(ds=case_when(.id=="data"~"data",TRUE~"simulation")) %>%
  ggplot(aes(x=year,y=pop,group=.id,color=ds))+
  geom_line()+
  labs(color="")