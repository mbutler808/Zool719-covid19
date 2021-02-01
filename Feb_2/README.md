# Feb 2 Exponential growth model and methods to obtain confidence intervals

1. 10:30am Randi will share her COVID19 paper

2. 11am Guest lecture

## Guest lecture by Lee Altenberg
[Dr. Altenberg](http://dynamics.org) will take us through computing confidence intervals on pandemic curves. He says Wikipedia level background is enough
Suggested Background Reading:  
* [Poisson Distribution](https://en.wikipedia.org/wiki/Poisson_distribution)  
* [Taylorʻs Law](https://en.wikipedia.org/wiki/Taylor%27s_law)

He will be using Mathematica, which you are welcome to use. We can follow also along in R using `expression( )` to hold our symbolic math and `eval()` to evaluate the math:
```
# expression( put math here )  
# eval( expression )

# for example  y = x^2 over the range x = 1 to 10
y <- expression( x^2 )
x <- 1:10
x
eval(y)
```

Got the simple example? Great. Now try:
```
#- Poisson Distribution----
## Has one parameter lambda
## Describes the probability that k events (infections) occur

## The probability mass function is given by:
poisson <- expression( (lambda^k * exp(-lambda))/ factorial(k) )

lambda <- 4
k <- 0:20
Fk <- eval(poisson)
plot(k, Fk, type="b")
```
And For Taylorʻs Law:
```
#- Taylorʻs Law-----
## Originally used to describe the spatial clustering of organisms
## in ecology, relating the variance and mean by a power law.
## Letʻs call our variance varT and mean meanT:

varT <- expression ( a*meanT^b )  
a <- 1
b <- 2
meanT <- 2:10
eval(varT)
```   


Next: Feb 9 at 11:00 Tom Blamey - Auto Regressive Model and Ensemble thoughts  
Back to [home](..)  
