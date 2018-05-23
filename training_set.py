pwr_data = { 'type' : 'pwr',
             'rxtrs' : ('ce14x14', 'ce16x16', 'w14x14', 'w15x15', 'w17x17',
                        's14x14', 's18x18', 'bw15x15', 'vver1000'),
             'enrich' : (1.5, 3.0, 4.0, 5.0),
             'burnup' : (3000, 6000, 9000, 12000, 15000, 18000,
                         21000, 24000, 27000, 30000, 33000, 36000, 39000,
                         42000, 45000, 48000, 51000, 54000, 57000, 60000,
                         63000) }

bwr_data = { 'type' : 'bwr',
             'rxtrs' : ('ge7x7-0', 'ge8x8-1', 'ge9x9-2', 'ge10x10-8', 
                        'abb8x8-1', 'atrium9x9-9', 'atrium10x10-9', 
                        'svea64-1', 'svea100-0'),
             'enrich' : (1.5, 3.0, 4.0, 5.0),
             'burnup' : (3000, 6000, 9000, 12000, 15000, 18000,
                         21000, 24000, 27000, 30000, 33000, 36000, 39000,
                         42000, 45000, 48000, 51000, 54000, 57000, 60000,
                         63000) }

vver_data = { 'type' : 'pwr',
              'rxtrs' : ('vver440',),
              'enrich' : (2.4, 3.6),
              'burnup' : (3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000,    
                         30000, 33000, 36000, 39000, 42000, 45000, 48000, 51000, 54000, 
                         57000, 60000, 63000)
              }

phwr_data = { 'type' : 'phwr',
              'rxtrs' : ('candu19', 'candu28', 'candu37'),
              'enrich' : (0.711,),
              'burnup' : (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 
                          6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500)
              }

cooling_intervals = (0.000694, 7, 30, 365.25, 2292) 

train_labels = []
for rxtr_data in [pwr_data, bwr_data, vver_data, phwr_data]:
    rxtr_type = rxtr_data['type']
    for rxtr in rxtr_data['rxtrs']:
        #print(rxtr_data['enrich'])
        for enrich in rxtr_data['enrich']:
            train_labels.append( {'ReactorType' : rxtr_type,
                                  'OrigenReactor' : rxtr,
                                  'Enrichment' : enrich,
                                  'Burnups' : rxtr_data['burnup'],
                                  'CoolingInts' : cooling_intervals } )



