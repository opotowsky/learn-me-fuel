########################################
########### TRAINING SET #1 ############
### Input parameters for ORIGEN sims ###
########################################

# This is Training Set #1, which is copied from the paper
# published by Dayman and Biegalski in 2013 (cited in prelim)

pwr_data = { 'type' : 'pwr',
             'rxtrs' : ('ce14x14', 'ce16x16', 'w14x14', 'w15x15', 'w17x17', 
                        's14x14', 'vver1000'),
             'enrich' : (2.8,),
             'arp_libs' : (('ce14_e30',), 
                           ('ce16_e30',), 
                           ('w14_e30',), 
                           ('w15_e30',), 
                           ('w17_e30',), 
                           ('s14_e30',), 
                           ('vver1000_e30',),
                           ),
             'burnup' : (600, 1550, 2500, 3450, 4400, 5350, 6300, 7250, 8200,
                         9150, 10100, 11050, 12000, 12950, 13900, 14850, 15800,
                         16750, 17700),
             # cooling time in days: 1 min, 1 wk, 1 mo, 1 yr 
             'cooling_intervals' : (0.000694, 7, 30, 365.25),
             'avg_power' : 32,
             'mod_density' : 0.72
             }

bwr_data = { 'type' : 'bwr',
             'rxtrs' : ('ge7x7-0', 'ge8x8-1', 'ge9x9-2', 'ge10x10-8', 
                        'abb8x8-1', 'atrium9x9-9', 'svea64-1', 'svea100-0'),
             'enrich' : (2.9,),
             'arp_libs' : (('ge7_e30w07',), 
                           ('ge8_e30w07',), 
                           ('ge9_e30w07',), 
                           ('ge10_e30w07',),
                           ('abb_e30w07',),
                           ('a9_e30w07',),
                           ('svea64_e30w07',), 
                           ('svea100_e30w07',),
                           ),
             'burnup' : (600, 1290, 1980, 2670, 3360, 4050, 4740, 5430, 6120,
                         6810, 7500, 8190, 8880, 9570, 10260, 10950, 11640,
                         12330),
             'cooling_intervals' : (0.000694, 7, 30, 365.25),
             'avg_power' : 23,
             'mod_density' : 0.72
             }

vver_data = { 'type' : 'pwr',
              'rxtrs' : ('vver440',),
              'enrich' : (3.6, 3.82, 4.25, 4.38),
              'arp_libs' : (('vver440_e36', 'vver440_rad_e38', 
                             'vver440_rad_e42', 'vver440_rad_e43'),
                            ),
              'burnup' : (600, 1550, 2500, 3450, 4400, 5350, 6300, 7250, 8200,
                          9150, 10100, 11050, 12000, 12950, 13900, 14850, 15800,
                          16750, 17700),
              'cooling_intervals' : (0.000694, 7, 30, 365.25),
              'avg_power' : 32,
              'mod_density' : 0.72
              }

phwr_data = { 'type' : 'phwr',
              'rxtrs' : ('candu28', 'candu37'),
              'enrich' : (0.711,),
              'arp_libs' : (('candu28_e07',), 
                            ('candu37_e07',),
                            ),
              'burnup' : (600, 1290, 1980, 2670, 3360, 4050, 4740, 5430, 6120,
                          6810, 7500, 8190, 8880, 9570, 10260, 10950, 11640,
                          12330),
              'cooling_intervals' : (0.000694, 7, 30, 365.25),
              'avg_power' : 22,
              'mod_density' : 0.84
              }

