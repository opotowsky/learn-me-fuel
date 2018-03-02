#!/usr/bin/python3

########################################
### Input parameters for ORIGEN sims ###
### for training set labels: burnup, ###
# cooling time, enrichment, rxtr type ##
########################################

NUM_CYCLES = 21
LWR_BURN = 3000
PHWR_BURN = 500

# order of O_RXTRS matters and all other ordering must follow
# hence, using tuples for everything

# ORIGEN reactor types + general types
O_RXTRS = ('ce14x14', 'ce16x16', 'w14x14', 'w15x15', 'w17x17', 
           's14x14', 's18x18', 'bw15x15', 'vver440', 'vver1000', 
           'ge7x7-0', 'ge8x8-1', 'ge9x9-2', 'ge10x10-8', 'abb8x8-1', 
           'atrium9x9-9', 'atrium10x10-9', 'svea64-1', 'svea100-0', 
           'candu19', 'candu28', 'candu37'
           )
RXTR_TYPES = ('pwr',)*10 + ('bwr',)*9 + ('phwr',)*3

# cooling time in days: 1 min, 1 wk, 1 mo, 1 yr, 8 yrs
COOLING_INTERVALS = (0.000694, 7, 30, 365.25, 2292) 

# enrichments %U235
LWR_ENR = (1.5, 3.0, 4.0, 5.0)
PHWR_ENR = (0.711,)
VVER_ENR = (2.4, 3.6)#, 3.82, 4.25, 4.38)
ENRICH = ((LWR_ENR,)*8 + (VVER_ENR,) + (LWR_ENR,)*10 + (PHWR_ENR,)*3)

# burnups (same for PWR/BWR)
LWRBURN = (3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000,    
           30000, 33000, 36000, 39000, 42000, 45000, 48000, 51000, 54000, 
           57000, 60000, 63000
           )
PHWRBURN = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 
            6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500
            )

# for this purpose we want just this in order, not separated tuples
RXTRLBL = (('pwr',)*len(LWR_ENR)*8 + ('pwr',)*len(VVER_ENR) + 
           ('pwr',)*len(LWR_ENR)*1 + ('bwr',)*len(LWR_ENR)*9 + 
           ('phwr',)*len(PHWR_ENR)*3
           )
ENRICHLBL = (LWR_ENR*8 + VVER_ENR + LWR_ENR*10 + PHWR_ENR*3)
BURNLBL = ((LWRBURN,)*len(LWR_ENR)*19 + (PHWRBURN,)*len(PHWR_ENR)*3)

rxtrlbl = list(RXTR_TYPES)
ORXTRLBL = []
for idx in range(0, len(rxtrlbl)):
    r_type = rxtrlbl[idx]
    o_rxtr = O_RXTRS[idx]
    if idx == 8:
        rxtrs = [o_rxtr,]*len(VVER_ENR)
    elif r_type == 'pwr':
        rxtrs = [o_rxtr,]*len(LWR_ENR)
    elif r_type == 'bwr':
        rxtrs = [o_rxtr,]*len(LWR_ENR)
    else:
        rxtrs = [o_rxtr,]*len(PHWR_ENR)
    ORXTRLBL = ORXTRLBL + rxtrs

# Dict for labelinig the training set using the simulation inputs
TRAIN_LABELS = {'ReactorType': RXTRLBL,
                'OrigenReactor': ORXTRLBL,
                'Enrichment': ENRICHLBL,
                'Burnup': BURNLBL,
                }

