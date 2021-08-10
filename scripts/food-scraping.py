# grab food from database and clean it
# cleaning it : putting it in a form compatible w/ mobile 

import os
import pandas as pd
import numpy as np

baseFolder = '../data/FoundationFoods'

data = dict()
for indx, file in enumerate(os.listdir(baseFolder)):
    if file[-4:] == '.csv':
        data[file[:-4]] = pd.read_csv(os.path.join(baseFolder, file)) 