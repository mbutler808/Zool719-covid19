# Mar 23 Partially Observed Markov Models (POMP)  

## Marguerite Butler 10:30am  

POMP is an R package used to simulate disease dynamics, combining compartmental models (SIR-type) and stochasticity in the form of partially observed Markov Models. It is extremely flexible and powerful.  

Please read the [Introduction to POMP](https://kingaa.github.io/pomp/vignettes/getting_started.html#Introduction) and work through the code.  

In class we will go over the [Ricker Model Example](https://kingaa.github.io/sbied/intro/ricker.html).  

Notes:  
* There are C code snippets used to speed up computation (and make some aspects of the coding cleaner).  
* Many parameters are obtained by simulation. Itʻs a tradeoff between realism and speed.
* Pay attention to the notation and the conventions.  
* POMP philosophy [here](https://kingaa.github.io/pomp/vignettes/oaxaca.html) Or you can watch this [YouTube](https://youtu.be/YZR5u_YedBY) - you can skip the long Q&A section in the mid-beginning, and the SIR details begin around 30min mark.  
* POMP frequently uses the tidyverse and ggplot, etc.  

Follow [Installation Instructions](https://kingaa.github.io/sbied/prep/)  

We will spend the class running the models and discussing how to apply it to problems of interest in Hawaii. Good examples [here](https://kingaa.github.io/pomp/docs.html) on Measles, Simulation-based inference, and model based inference. There is an Ebola example as well under Sumulation-based inference.  

Recent paper using POMP to model SARS-Cov-2  
[Kain et al., 2021](./KainChildBeckerMordecai2021.pdf) Chopping the tail: How preventing superspreading can help to maintain COVID-19 control. Epidemics 34: 100430 https://doi.org/10.1016/j.epidem.2020.100430

Language: R  
Libraries:  
  * pomp
  * tidyverse
  * ggplot


Next: March 30 What is our plan? Please send me your preferred dates.  

Back to [home](..)  
