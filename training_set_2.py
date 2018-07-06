########################################
########### TRAINING SET #2 ############
### Input parameters for ORIGEN sims ###
########################################

# This is Training Set #2, which is based on the breadth 
# of params from the SFCOMPO database

pwr_data = { 'type' : 'pwr',
             'rxtrs' : ('ce14x14', 'ce16x16', 'w14x14', 'w15x15', 'w17x17',
                        's14x14', 's18x18', 'bw15x15', 'vver1000'),
             'enrich' : (1.5, 3.0, 4.0, 5.0),
             'arp_libs' : (('ce14_e15', 'ce14_e30', 'ce14_e40', 'ce14_e50'),
                           ('ce16_e15', 'ce16_e30', 'ce16_e40', 'ce16_e50'), 
                           ('w14_e15', 'w14_e30', 'w14_e40', 'w14_e50'), 
                           ('w15_e15', 'w15_e30', 'w15_e40', 'w15_e50'), 
                           ('w17_e15', 'w17_e30', 'w17_e40', 'w17_e50'), 
                           ('s14_e15', 's14_e30', 's14_e40', 's14_e50'), 
                           ('s18_e15', 's18_e30', 's18_e40', 's18_e50'), 
                           ('bw15_e15', 'bw15_e30', 'bw15_e40', 'bw15_e50'),
                           ('vver1000_e15', 'vver1000_e30', 'vver1000_e40', 'vver1000_e50'),
                           ),
             'burnup' : (3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 
                         27000, 30000, 33000, 36000, 39000, 42000, 45000, 48000, 
                         51000, 54000, 57000, 60000, 63000), 
             # cooling time in days: 1 min, 1 wk, 1 mo, 1 yr, 8 yrs, 
             'cooling_intervals' : (0.000694, 7, 30, 365.25, 2292),
             'avg_power' : 30,
             'mod_density' : 0.72
             }

bwr_data = { 'type' : 'bwr',
             'rxtrs' : ('ge7x7-0', 'ge8x8-1', 'ge9x9-2', 'ge10x10-8', 
                        'abb8x8-1', 'atrium9x9-9', 'atrium10x10-9', 
                        'svea64-1', 'svea100-0'),
             'enrich' : (1.5, 3.0, 4.0, 5.0),
             'arp_libs' : (('ge7_e15w07', 'ge7_e30w07', 'ge7_e40w07', 'ge7_e50w07'),
                           ('ge8_e15w07', 'ge8_e30w07', 'ge8_e40w07', 'ge8_e50w07'), 
                           ('ge9_e15w07', 'ge9_e30w07', 'ge9_e40w07', 'ge9_e50w07'), 
                           ('ge10_e15w07', 'ge10_e30w07', 'ge10_e40w07', 'ge10_e50w07'), 
                           ('abb_e15w07', 'abb_e30w07', 'abb_e40w07', 'abb_e50w07'), 
                           ('a9_e15w07', 'a9_e30w07', 'a9_e40w07', 'a9_e50w07'), 
                           ('a10_e15w07', 'a10_e30w07', 'a10_e40w07', 'a10_e50w07'), 
                           ('svea64_e15w07', 'svea64_e30w07', 'svea64_e40w07', 'svea64_e50w07'), 
                           ('svea100_e15w07', 'svea100_e30w07', 'svea100_e40w07', 'svea100_e50w07'),
                           ), 
             'burnup' : (3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 
                         27000, 30000, 33000, 36000, 39000, 42000, 45000, 48000, 
                         51000, 54000, 57000, 60000, 63000), 
             'cooling_intervals' : (0.000694, 7, 30, 365.25, 2292),
             'avg_power' : 20,
             'mod_density' : 0.72
             }

vver_data = { 'type' : 'pwr',
              'rxtrs' : ('vver440',),
              'enrich' : (1.6, 2.4, 3.6, 3.82, 4.25, 4.38),
              'arp_libs' : (('vver440_e16', 'vver440_e24', 'vver440_e36', 
                             'vver440_rad_e38', 'vver440_rad_e42', 'vver440_rad_e43'),
                            ),
              'burnup' : (3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 
                          27000, 30000, 33000, 36000, 39000, 42000, 45000, 48000, 
                          51000, 54000, 57000, 60000, 63000),
              'cooling_intervals' : (0.000694, 7, 30, 365.25, 2292),
              'avg_power' : 30,
              'mod_density' : 0.72
              }

phwr_data = { 'type' : 'phwr',
              'rxtrs' : ('candu19', 'candu28', 'candu37'),
              'enrich' : (0.711,),
              'arp_libs' : (('candu19_e07',), 
                            ('candu28_e07',), 
                            ('candu37_e07',)
                            ),
              'burnup' : (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 
                          5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 
                          9500, 10000, 10500),
              'cooling_intervals' : (0.000694, 7, 30, 365.25, 2292),
              'avg_power' : 20,
              'mod_density' : 0.84
              }

