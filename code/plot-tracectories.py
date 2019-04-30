

import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import sys
# from .data.regresstools import lowess
from datetime import datetime
import time
import pickle
import pandas as pd

%matplotlib qt5

homechar = "C:\\"

# day = "April10"
day = "April11"

vid = "pi71_1554994358" # ?
# vid = "pi71_1554994823" # 70% flow?
# vid = "pi71_1554995860" # 70?

inptfile = os.path.join(homechar, "Projects", "ptv-aquatron2019", "data", "processed", vid + ".pkl")

infile = open(inptfile,'rb')
data = pickle.load(infile)
infile.close()
