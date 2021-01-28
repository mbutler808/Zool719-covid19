import numpy as np
import scipy.integrate as integrate
from datetime import datetime, timedelta

def run_scenario(Model_Begin,
                 Model_End,
                 Starting_Hospitalizations,
                 Starting_ICU,
                 Starting_Fatal,
                 Size_of_population,
                 Initial_Infected,
                 Latent_period,
                 Infectious_period,
                 Home_to_Hosp_lag,
                 Length_of_Hospitalization,
                 Length_of_ICU,
                 Perc_Hosp,
                 Perc_Hosp_to_ICU,
                 Perc_ICU_to_Hosp,
                 Rt_array,
                 Travel_array):

    # ~~~~ MODEL PARAMETERS ~~~~
    # Set initial state & parameter settings

    # Set Initial Population
    N                           = Size_of_population # Size of population
    I0                          = Initial_Infected   # Number of initial infections

    # Set Durations to determine how long a person stays in a given compartment
    D_latent                    = Latent_period       # Length of incubation period (days) # !age distributable
    D_infectious                = Infectious_period       # Duration patient is infectious (days)
    D_hospital_lag              = Home_to_Hosp_lag       # Time delay for severe cases (non-icu) to be hospitalized
    D_length_of_hospitalization = Length_of_Hospitalization       # Length of hospital stay (recovery time for severe cases)
    D_length_of_ICU             = Length_of_ICU      # Length of hospital stay for ICU cases

    # Set factors for flow between compartment
    p_hosp                      = Perc_Hosp    # Percent of infectious people who go to hospital
    p_H_C                       = Perc_Hosp_to_ICU     # Percent of Hospitalizations that flow to ICU (remainder go to Recovered (R))
    p_C_H                       = Perc_ICU_to_Hosp     # Percent of ICU that flow to back to Hospitalizations (remainder go to Fatal (F))

    def ndays(date1, date2):
        date_format = "%m/%d/%Y"
        date1 = datetime.strptime(date1, date_format)
        date2 = datetime.strptime(date2, date_format)
        delta = date2 - date1
        return delta.days

    difference_in_days = ndays(Model_Begin, Model_End)

    # ~~~~ R(t) ~~~~
    # Returns reproduction number as a function of time
    def Rt(Rt_array, t):
        max_days = 0
        R_value = 0

        # Provide the correct Rt for the given t
        for idx in range(len(Rt_array)):
            if t >= Rt_array[idx][0] and Rt_array[idx][0] >= max_days:
                R_value = Rt_array[idx][1]
                max_days = Rt_array[idx][0]
        return R_value

    # ~~~~ TRAVEL CASES ~~~~
    # Update case counts to include incoming cases from travel
    def travel_cases(t) :
        new_E = 0 # New number to add to Exposed compartment
        new_I = 0 # New number to add to Infectious compartment

        # Provide the correct Rt for the given t
        for idx in reversed(range(len(Travel_array))):
            # If current time has passed the date, add new cases
            if t >= Travel_array[idx][0]:
                # Update E and I numbers to reflect travel additions
                new_E += Travel_array[idx][1]
                new_I += Travel_array[idx][2]
                # Remove element from array
                Travel_array.pop(idx)
        return [new_E, new_I]

    def toDate(t):
        starting_datetime = datetime.strptime(Model_Begin, "%m/%d/%Y")
        return (starting_datetime + timedelta(days=t)).strftime("%m/%d/%Y")

    # ~~~~ DIFFEQs ~~~~
    # Defines DiffEqs that determine flow between compartments
    # Returns a vector used to store compartment numbers
    def f(t, x):
        beta = Rt(Rt_array, t)/(D_infectious)
        a     = 1/D_latent
        gamma = 1/D_infectious

        S        = x[0] # Susceptible
        E        = x[1] # Exposed
        I        = x[2] # Infectious
        R        = x[3] # Recovering (R)
        D        = x[4] # Delayed (to Hospitalization)
        H        = x[5] # Hospitalization
        C        = x[6] # ICU
        F        = x[7] # Fatal
        N        = x[8] # New Hospitalizations


        dS        = -beta*I*S
        dE        =  beta*I*S - a*E
        dI        =  a*E - gamma*I
        dR        =  (1 - p_hosp)*gamma*I + (1 - p_H_C)*((1/D_length_of_hospitalization)*H) # people not going to hospital + those that get out of the hospital
        dD        =  (p_hosp)*gamma*I - (1/D_hospital_lag)*D
        dH        =  (1/D_hospital_lag)*D - (1/D_length_of_hospitalization)*H + p_C_H*((1/D_length_of_ICU)*C)
        dC        =  p_H_C*((1/D_length_of_hospitalization)*H)  - ((1/D_length_of_ICU)*C)
        dF        =  (1 - p_C_H)*((1/D_length_of_ICU)*C)
        dN        =  (1/D_hospital_lag)*D

        #       0   1   2   3      4        5          6      7      8
        return [dS, dE, dI, dR,    dD,      dH,        dC,    dF,    dN]

    # Vector to store all the compartment numbers (in fractions to add up to 1 for total population)
    v = np.array([1 - I0/N, 0, I0/N, 0, 0, Starting_Hospitalizations/N, Starting_ICU/N, Starting_Fatal/N, 0])

    # ~~~~ TIME ~~~~
    # Array to keep logs of all the recorded compartment numbers
    Output_array = []

    # print("Starting simulation")
    # ~~~~ RUN SIMULATION ~~~~
    for t in range(difference_in_days+1):
        # Add travel cases if reached point
        new_travel_cases = travel_cases(t)
        v[1] += new_travel_cases[0] / N # Add Exposed cases from travel
        v[2] += new_travel_cases[1] / N # Add Infectious cases from travel

        Output_array.append([
            toDate(t), # Date
            N * v[2], # Infected
            N * v[5], # Hosp
            N * v[6], # ICU
            N * v[7], # Fatal
            Rt(Rt_array, t), # Rt
            new_travel_cases[0], # Travel E
            new_travel_cases[1], # Travel I
            N * v[0], # Susceptible
            N * (v[2]+v[3]+v[4]+v[5]+v[6]+v[7]), # Total infected
            N * v[8]
        ])

        # Simulate compartment numbers for next time step
        update_v = integrate.solve_ivp(fun=f, t_span=(t, t+1), y0=v, t_eval=[t+1])
        v = np.transpose(update_v['y'])[0]


    # print("Finished simulation")

    return Output_array
