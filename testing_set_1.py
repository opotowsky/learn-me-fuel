########################################
########### TESTING SET #1 #############
### Input parameters for ORIGEN sims ###
########################################

# This is Testing Set #1, which is copied from the paper
# published by Dayman and Biegalski in 2013 (cited in prelim)

cool1 = (0.000694, 7, 30, 365.25) #1 min, 1 week, 1 month, 1 year in days
cool2 = (0.002082, 9, 730.5) #3 min, 9 days, 2 years in days
cool3 = (7, 9) #7 and 9 days

pwr_data1 = { 'type' : 'pwr',
              'rxtrs' : ('ce16x16',),
              'enrich' : (2.8,),
              'arp_libs' : (('ce16_e30',),),
              'burnup' : (1700, 8700, 17000), 
              'cooling_intervals' : cool1,
              'avg_power' : 32,
              'mod_density' : 0.72
             }

pwr_data2 = { 'type' : 'pwr',
              'rxtrs' : ('ce16x16',),
              'enrich' : (2.8,),
              'arp_libs' : (('ce16_e30',),),
              'burnup' : (8700, 9150), 
              'cooling_intervals' : cool2,
              'avg_power' : 32,
              'mod_density' : 0.72
             }

pwr_data3 = { 'type' : 'pwr',
              'rxtrs' : ('ce16x16',),
              'enrich' : (3.1,),
              'arp_libs' : (('ce16_e30',),),
              'burnup' : (8700, 9150), 
              'cooling_intervals' : cool3,
              'avg_power' : 32,
              'mod_density' : 0.72
             }

bwr_data1 = { 'type' : 'bwr',
              'rxtrs' : ('ge7x7-0',),
              'enrich' : (2.9,),
              'arp_libs' : (('ge7_e30w07',),),
              'burnup' : (2000, 7200, 10800), 
              'cooling_intervals' : cool1,
              'avg_power' : 23,
              'mod_density' : 0.72
             }

bwr_data2 = { 'type' : 'bwr',
              'rxtrs' : ('ge7x7-0',),
              'enrich' : (2.9,),
              'arp_libs' : (('ge7_e30w07',),),
              'burnup' : (7200, 8800), 
              'cooling_intervals' : cool2,
              'avg_power' : 23,
              'mod_density' : 0.72
             }

bwr_data3 = { 'type' : 'bwr',
              'rxtrs' : ('ge7x7-0',),
              'enrich' : (3.2,),
              'arp_libs' : (('ge7_e30w07',),),
              'burnup' : (7200, 8800), 
              'cooling_intervals' : cool3,
              'avg_power' : 23,
              'mod_density' : 0.72
             }

phwr_data1 = { 'type' : 'phwr',
              'rxtrs' : ('candu28',),
              'enrich' : (0.711,),
              'arp_libs' : (('candu28_e07',),),
              'burnup' : (1400, 5000, 11000), 
              'cooling_intervals' : cool1,
              'avg_power' : 22,
              'mod_density' : 0.84
              }

phwr_data2 = { 'type' : 'phwr',
              'rxtrs' : ('candu28',),
              'enrich' : (0.711,),
              'arp_libs' : (('candu28_e07',),),
              'burnup' : (5000, 6120), 
              'cooling_intervals' : cool2,
              'avg_power' : 22,
              'mod_density' : 0.84
              }

testing_set = [pwr_data1, pwr_data2, pwr_data3, 
               bwr_data1, bwr_data2, bwr_data3, 
               phwr_data1, phwr_data2]
