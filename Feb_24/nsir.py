__author__ = 'szapudi'
"""
   network science inspired generalization of the SIR equations

Wed Jun 10 14:13:36 HST 2020
   (c) Istvan Szapudi
"""
import numpy as N

#SIR model: I modify
#https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
class SIRParams():
    """
    container object for fundamental parmameters of the simple SIR model
    """
    # Total population, N.
    nPop = 1000.
    # to compare with https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html
    # top figure: Japan
    #nPop = 15000.
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 1., 0.
    # to compare with https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html
    #I0, R0 = 2., 0.
    # Everyone else, S0, is susceptible to infection initially.
    S0 = nPop - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    # physical meaning: 1/beta = T_c typical time between contacts
    #                   1/gamma = T_r time for recovery
    #                   R_0 = beta/gamma = T_r/T_c = number of people infected on average
    beta, gamma = 0.2, 1. / 10
    # to compare with https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html
    #beta, gamma = 0.00002856*nPop, 0.29819303
    # checks out Wed Jul  1 16:09:41 HST 2020
    # A grid of time points (in days)
    t = N.linspace(0, 160, 160)


# The SIR model differential equations.
def derivSIR(y, t, nPop, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / nPop
    dIdt = beta * S * I / nPop - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def plotSIR():
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    # Initial conditions vector
    sp = SIRParams()
    y0 = sp.S0, sp.I0, sp.R0
    t = sp.t
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(derivSIR, y0, t, args=(sp.nPop, sp.beta, sp.gamma))
    S, I, R = ret.T

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S/sp.nPop, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/sp.nPop, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/sp.nPop, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
   # for spine in ('top', 'right', 'bottom', 'left'):
   #     ax.spines[spine].set_visible(False)
    plt.show()

class NSIRParams():
    """
    container object for fundamental parmameters of the simple NSIR model
    """
    def __init__(self,sus0,inf0,rec0,beta = 0.2, gamma = 0.1, tmax = 160):
        """
         the main initial arrays, sus0[k] = number of susceptibles with k network connections
         typically, initially most people are susceptible with sus0[k] \propto k^-(power) where power = 2-3
         rec0 is zero and a small seed of infection is in inf0
         the three main variables must have the same length
        """
        self.sus0 = sus0 #suceptible
        self.inf0 = inf0 #infected
        self.rec0 = rec0 #recovered
        # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
        # physical meaning: 1/beta = T_c typical time between contacts
        #                   1/gamma = T_r time for recovery
        #                   R_0 = beta/gamma = T_r/T_c = number of people infected on average
        self.beta, self.gamma = beta, gamma
        # A grid of time points (in days)
        t = N.linspace(0, tmax, tmax+1)

        self.bins = self.sus0.size
        self.n = self.sus0+self.inf0+self.rec0
        self.links = N.sum(self.n*N.arange(self.bins))

        self.t = N.linspace(0, tmax, tmax+1)

        #effective beta0 and R0
        self.infLinks = N.sum(self.inf0 * N.arange(self.bins))  # total number of links of the infected
        self.infP = self.beta * self.infLinks / self.links  # the probability of infection through a random link
        self.pk = 1.0 - N.exp(N.arange(self.bins) * N.log(1 - self.infP))  # p(k) = 1-(1-pLink)^k infection probability for a vertex with k links

        self.betaEff0 = N.sum(self.pk*self.sus0)*N.sum(self.n)/N.sum(self.inf0)/N.sum(self.sus0)
        self.R0 = self.betaEff0/self.gamma/N.sum(self.n)*N.sum(self.sus0)

def calcEffR(S,I,sp):
    """
    calculate the effective beta and R = beta/gamma as a function of time
    k x t vectors
    :param S: susceptibles
    :param I: infected
    :param sp: parameter object
    :return: betaEff[:], R[:]
    """
    infLinks = N.sum(I * N.arange(sp.bins), axis=1)  # total number of links of the infected @ t
    infP = sp.beta * infLinks / sp.links  # the probability of infection through a random link @ t
    pk = 1.0 - N.exp(
        N.outer(N.log(1 - infP), N.arange(sp.bins)))  # p(k) = 1-(1-pLink)^k infection probability for a vertex with k links
    betaEff = N.sum(pk * S, axis=1) * N.sum(sp.n) / N.sum(I,axis=1) / N.sum(S,axis=1) #betaeff @t

    Rt = betaEff / sp.gamma/N.sum(sp.n) * N.sum(S,axis=1)
    return infLinks, infP, pk, betaEff, Rt


def derivNSIR(y, t, nsp):
    """
    :param y:  = sus, inf == susceptibles, infected
    :param t: timesteps to be calculated (in days)
    :param nsp: nsir parameter object
    :return: dy/dt
    note that we are not calculating the recovered fraction, as they are always R = N-S-I
    """
    sus = y[:nsp.bins]
    inf = y[nsp.bins:2*nsp.bins]

    infLinks = N.sum(inf*N.arange(nsp.bins)) # total number of links of the infected
    infP = nsp.beta * infLinks/nsp.links # the probability of infection through a random link

    pk = 1.0-N.exp(N.arange(nsp.bins) * N.log(1-infP)) #p(k) = 1-(1-pLink)^k infection probability for a vertex with k links

    dSus = -pk*sus
    dInf =  pk*sus - nsp.gamma*inf

    return N.concatenate((dSus,dInf))

def genPowerLaw(bins = 10,ntot = 1000,gamma = 2.5):
    """
    generate a network science motivated power link distribution
    :param bins: number of bins of the distribution
    :param ntot: total number of people
    :param gamma: slope n(k) \propto k^-gamma, n[0] = 0
    :return: n(k) an array
    """
    a = N.arange(bins)
    b = N.zeros(bins)
    b[1:] = N.exp(-N.log(a[1:]) * gamma)

    return ntot*b/sum(b)

def testNSIR():
    """
    test the NSIR integrator with the simplest case SIR limit with k=1 nodes only
    :return: Rt
    """
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    # Initial conditions vector
    sus0 = N.zeros(10)
    inf0 = N.zeros(10)
    rec0 = N.zeros(10)
    sus0[1] = 1000
    inf0[1] = 1
    sp = NSIRParams(sus0,inf0,rec0)
    y0 = N.concatenate((sp.sus0, sp.inf0))
    t = sp.t
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(derivNSIR, y0, t, args=(sp,))
    S = ret[:,1]
    I = ret[:,11]
    R = (-S - I + sum(sp.n))

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Fraction')
    ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
  #  for spine in ('top', 'right', 'bottom', 'left'):
  #      ax.spines[spine].set_visible(False)
    plt.show()
    return sp.R0*S/N.sum(sp.n)

def plotNSIR(b = 30, ntot=100000, infectBin = 10, beta = 0.07,gamma = 2.5):
    """
    plot the NSIR models
    :return: S, I distributions, sp: parameter container class
    """
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    # Initial conditions vector
    sus0 = genPowerLaw(bins = b, gamma=gamma, ntot=ntot)
    inf0 = N.zeros(b)
    rec0 = N.zeros(b)
    inf0[infectBin] = 1 #infect a hub node
    sp = NSIRParams(sus0,inf0,rec0,beta=beta)
    y0 = N.concatenate((sp.sus0, sp.inf0))
    t = sp.t
    # renormalized beta
    print(sp.betaEff0, sp.R0)
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(derivNSIR, y0, t, args=(sp,))
    S = N.sum(ret[:,:b],axis=1)
    I = N.sum(ret[:,b:],axis=1)
    R = (-S - I + sum(sp.n))

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S/ntot, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/ntot, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/ntot, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Fraction')
    ax.set_ylim(0,1.2)
#    ax.yaxis.set_tick_params(length=0)
#    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
 #   for spine in ('top', 'right', 'bottom', 'left'):
 #       ax.spines[spine].set_visible(True)
    plt.show()
    return ret[:,:b], ret[:,b:], sp

def plotFidvsNSIR(b = 30, ntot=100000, infectBin = 10, beta = 0.07,gamma = 2.5):
    """
    plot a fiducial plot vs parameters changed
    :return: S, I distributions, sp: parameter container class of the second choice
    """
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt

    # FIDUCIAL CASE
    # Initial conditions vector
    bfid = 30
    sus0 = genPowerLaw(bins = bfid, gamma=2.5, ntot=100000)
    inf0 = N.zeros(bfid)
    rec0 = N.zeros(bfid)
    inf0[10] = 1 #infect a hub node
    sp = NSIRParams(sus0,inf0,rec0,beta=beta)
    y0 = N.concatenate((sp.sus0, sp.inf0))
    t = sp.t
    # renormalized beta
#    print(sp.betaEff0, sp.R0)
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(derivNSIR, y0, t, args=(sp,))
    S = N.sum(ret[:,:bfid],axis=1)
    I = N.sum(ret[:,bfid:],axis=1)
    R = (-S - I + sum(sp.n))

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S/ntot, 'b', alpha=0.5, linestyle='dotted', lw=2)
    ax.plot(t, I/ntot, 'r', alpha=0.5, linestyle='dotted', lw=2)
    ax.plot(t, R/ntot, 'g', alpha=0.5, linestyle='dotted', lw=2)
#    ax.set_xlabel('Time /days')
#    ax.set_ylabel('Fraction')
#    ax.set_ylim(0,1.2)
#    ax.yaxis.set_tick_params(length=0)
#    ax.xaxis.set_tick_params(length=0)
#    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
#    legend = ax.legend()
#    legend.get_frame().set_alpha(0.5)

    sus0 = genPowerLaw(bins=b, gamma=gamma, ntot=ntot)
    inf0 = N.zeros(b)
    rec0 = N.zeros(b)
    inf0[infectBin] = 1  # infect a hub node
    sp = NSIRParams(sus0, inf0, rec0, beta=beta)
    y0 = N.concatenate((sp.sus0, sp.inf0))
    t = sp.t
    # renormalized beta
    #    print(sp.betaEff0, sp.R0)
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(derivNSIR, y0, t, args=(sp,))
    S = N.sum(ret[:, :b], axis=1)
    I = N.sum(ret[:, b:], axis=1)
    R = (-S - I + sum(sp.n))

    # Plot the data on three separate curves for S(t), I(t) and R(t)
 #   fig = plt.figure(facecolor='w')
 #   ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S / ntot, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I / ntot, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R / ntot, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Fraction')
    ax.set_ylim(0, 1.2)

    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)

 #   for spine in ('top', 'right', 'bottom', 'left'):
 #       ax.spines[spine].set_visible(True)
    plt.show()
    return ret[:,:b], ret[:,b:], sp

def plotSIRvsNSIR(b = 30, ntot=100000, infectBin = 10, beta = 0.07,gamma = 2.5):
    """
    compare the SIR and the NSIR models
    :return: S, I distributions, sp: parameter container class of the second choice
    """
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt

    # SIR CASE
    # Initial conditions vector
    sus0 = N.zeros(10)
    inf0 = N.zeros(10)
    rec0 = N.zeros(10)
    sus0[1] = 1000
    inf0[1] = 1
    sp = NSIRParams(sus0,inf0,rec0)
    y0 = N.concatenate((sp.sus0, sp.inf0))
    t = sp.t
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(derivNSIR, y0, t, args=(sp,))
    S = ret[:,1]
    I = ret[:,11]
    R = (-S - I + sum(sp.n))

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S/1000, 'b', alpha=0.5, linestyle='dotted', lw=2)
    ax.plot(t, I/1000, 'r', alpha=0.5, linestyle='dotted', lw=2)
    ax.plot(t, R/1000, 'g', alpha=0.5, linestyle='dotted', lw=2)
#    ax.set_xlabel('Time /days')
#    ax.set_ylabel('Fraction')
#    ax.set_ylim(0,1.2)
#    ax.yaxis.set_tick_params(length=0)
#    ax.xaxis.set_tick_params(length=0)
#    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
#    legend = ax.legend()
#    legend.get_frame().set_alpha(0.5)

    sus0 = genPowerLaw(bins=b, gamma=gamma, ntot=ntot)
    inf0 = N.zeros(b)
    rec0 = N.zeros(b)
    inf0[infectBin] = 1  # infect a hub node
    sp = NSIRParams(sus0, inf0, rec0, beta=beta)
    y0 = N.concatenate((sp.sus0, sp.inf0))
    t = sp.t
    # renormalized beta
    #    print(sp.betaEff0, sp.R0)
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(derivNSIR, y0, t, args=(sp,))
    S = N.sum(ret[:, :b], axis=1)
    I = N.sum(ret[:, b:], axis=1)
    R = (-S - I + sum(sp.n))

    # Plot the data on three separate curves for S(t), I(t) and R(t)
 #   fig = plt.figure(facecolor='w')
 #   ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S / ntot, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I / ntot, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R / ntot, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Fraction')
    ax.set_ylim(0, 1.2)

    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)

 #   for spine in ('top', 'right', 'bottom', 'left'):
 #       ax.spines[spine].set_visible(True)
    plt.show()
    return ret[:,:b], ret[:,b:], sp

def plotRts():
    """
    figure showing the Rt behavior as a function of infectBin
    :return: none
    """
    import matplotlib.pyplot as plt

    rtSIR = testNSIR()
    S,I,sp = plotFidvsNSIR(infectBin=29)
    infLinks, infP, pk, betaEff, rt29 = calcEffR(S,I,sp)
    S,I,sp = plotFidvsNSIR()
    infLinks, infP, pk, betaEff, rt10 = calcEffR(S,I,sp)
    S,I,sp = plotFidvsNSIR(infectBin=2)
    infLinks, infP, pk, betaEff, rt2 = calcEffR(S,I,sp)

    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(rtSIR, 'k', alpha=0.5, linestyle='dotted', lw=2, label='SIR')
    ax.plot(rt10, 'k', alpha=0.5, linestyle='--', lw=2, label='NSIR')
    ax.plot(rt29, 'b', alpha=0.5, lw=2, label = 'NSIR top')
    ax.plot(rt2, 'r', alpha=0.5, lw=2, label = 'NSIR bottom')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('$R_t$')
    ax.set_ylim(0, 8)

    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.show()

def plotS(b=30, ntot=100000, infectBin=10, beta=0.07, gamma=2.5):
    """
    plot all the infected bins
    :return: S, I distributions, sp: parameter container class of the second choice
    """
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt

    # FIDUCIAL CASE
    # Initial conditions vector
    bfid = 30
    sus0 = genPowerLaw(bins=bfid, gamma=2.5, ntot=100000)
    inf0 = N.zeros(bfid)
    rec0 = N.zeros(bfid)
    inf0[10] = 1  # infect a hub node
    sp = NSIRParams(sus0, inf0, rec0, beta=beta)
    y0 = N.concatenate((sp.sus0, sp.inf0))
    t = sp.t
    # renormalized beta
    #    print(sp.betaEff0, sp.R0)
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(derivNSIR, y0, t, args=(sp,))
    S = N.sum(ret[:, :bfid], axis=1)
    I = N.sum(ret[:, bfid:], axis=1)
    R = (-S - I + sum(sp.n))

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.semilogy(t, S / ntot, 'b', alpha=0.5, linestyle='dotted', lw=2)
    ax.semilogy(t, I / ntot, 'r', alpha=0.5, linestyle='dotted', lw=2)
    ax.semilogy(t, R / ntot, 'g', alpha=0.5, linestyle='dotted', lw=2)
    #    ax.set_xlabel('Time /days')
    #    ax.set_ylabel('Fraction')
    #    ax.set_ylim(0,1.2)
    #    ax.yaxis.set_tick_params(length=0)
    #    ax.xaxis.set_tick_params(length=0)
    #    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    #    legend = ax.legend()
    #    legend.get_frame().set_alpha(0.5)


    #    individual infected cases
    sus0 = genPowerLaw(bins=b, gamma=gamma, ntot=ntot)
    inf0 = N.zeros(b)
    rec0 = N.zeros(b)
    inf0[infectBin] = 1  # infect a hub node
    sp = NSIRParams(sus0, inf0, rec0, beta=beta)
    y0 = N.concatenate((sp.sus0, sp.inf0))
    t = sp.t
    # renormalized beta
    #    print(sp.betaEff0, sp.R0)
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(derivNSIR, y0, t, args=(sp,))
    I = ret[:, b:]
    for i in range(b):
        ax.semilogy(t, I[:,i] / ntot, 'r', alpha=0.5, lw=2,)

    ax.set_xlabel('Time /days')
    ax.set_ylabel('Fraction')
    ax.set_ylim(0.00000001, 1.2)

    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)

    #   for spine in ('top', 'right', 'bottom', 'left'):
    #       ax.spines[spine].set_visible(True)
    plt.show()
    return ret[:, :b], ret[:, b:], sp