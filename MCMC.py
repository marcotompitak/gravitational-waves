# This script runs, non-interactively, the MCMC simulation
# also presented in the accompanything iPython notebook. It
# was used to generate the data available for loading in.

import numpy as np
import pymc
import math
import pandas as pd
np.random.seed(0) # For reproducibility

#def simple_signal(t, offset, A, f, df):
#    return offset + A * np.sin(f * t + df * t * t)

# We will use the same chirp function as previously
def chirp(t, tc, offset, A, dA, f, df):
    """
    Generate a chirp signal.
    
    Arguments:
    t      -- array-like containing the times at which to evaluate the signal.
    tc     -- time of coalescence, after which the signal is terminated; for times beyond tc the signal is set to the constant background.
    offset -- constant background term
    A      -- initial signal amplitude at t=0
    dA     -- linear coefficient describing the increase of the amplitude with time
    f      -- initial signal frequency at t=0
    df     -- linear coefficient describing the increase of the frequency with time
    """
    chirp = offset + (A + dA*t) * np.sin(2*math.pi*(f + df*t) * t)
    chirp[t>tc] = offset
    return chirp

# Let's choose some values for our injected signal
tc_true = 75
offset_true = 30
A_true = 6
dA_true = 0.05
f_true = 0.2
df_true = 0.007

# Noise strength; we will keep it low here so we 
# don't have to run our sampling for too long
sigma = 50

# Time axis
t = np.linspace(0,100,10001)

#y_simple_true = simple_signal(t, offset_true, A_true, f_true, df_true)
#y_simple_obs = np.random.normal(y_simple_true, sigma)

# Injecting our signal into the noise
y_true = chirp(t, tc_true, offset_true, A_true, dA_true, f_true, df_true)
y_obs = np.random.normal(y_true, sigma)

# Helper data to keep track of everything
parameters = ['tc', 'offset', 'A', 'dA', 'f', 'df']
bounds = {'tc' : (0,100), 'offset' : (0,100), 'A' : (0,10), 'dA' : (0,0.1), 'f' : (0,1), 'df' : (0,0.1)}

# Defining our parameters using pymc.Uniform()
offset = pymc.Uniform(
    'offset', \
    bounds['offset'][0], \
    bounds['offset'][1], \
    value = bounds['offset'][0] + \
    (bounds['offset'][1]-bounds['offset'][0])*np.random.random()
)

tc = pymc.Uniform(
    'tc', \
    bounds['tc'][0], \
    bounds['tc'][1], \
    value = bounds['tc'][0] + \
    (bounds['tc'][1]-bounds['tc'][0])*np.random.random()
)

A = pymc.Uniform(
    'A', \
    bounds['A'][0], \
    bounds['A'][1], \
    value = bounds['A'][0] + \
    (bounds['A'][1]-bounds['A'][0])*np.random.random()
)

dA = pymc.Uniform(
    'dA', \
    bounds['dA'][0], \
    bounds['dA'][1], \
    value = bounds['dA'][0] + \
    (bounds['dA'][1]-bounds['dA'][0])*np.random.random()
)

f = pymc.Uniform(
    'f', \
    bounds['f'][0], \
    bounds['f'][1], \
    value = bounds['f'][0] + \
    (bounds['f'][1]-bounds['f'][0])*np.random.random()
)

df = pymc.Uniform(
    'df', \
    bounds['df'][0], \
    bounds['df'][1], \
    value = bounds['df'][0] + \
    (bounds['df'][1]-bounds['df'][0])*np.random.random()
)

@pymc.deterministic
def y_model(t=t, tc=tc, offset=offset, A=A, dA=dA, f=f, df=df):
    return chirp(t, tc, offset, A, dA, f, df)

y = pymc.Normal('y', mu=y_model, tau=sigma**-2, observed=True, value=y_obs)

model = pymc.Model([y, tc, offset, A, dA, f, df, y_model])
M = pymc.MCMC(model)

M.sample(iter=100000, burn=50000, thin=1)

def mcmc_dataframe(M, skip_vars):
    names = []
    data = []
    for var in M.variables:
        try:
            name = var.trace.name
            if name not in skip_vars:
                names.append(var.trace.name)
                data.append(var.trace.gettrace().tolist())
        except AttributeError:
            pass
    df = pd.DataFrame(data).T
    df.columns = names
    return df

# Save as csv
mcmc_dataframe(M, ['y_model']).to_csv('samples_pregen.csv', index = False)