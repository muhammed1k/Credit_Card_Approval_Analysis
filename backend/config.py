import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

#loading data Files
def load_data(path, file_names):
    application_details = pd.read_csv(path / file_names[0])
    application_history = pd.read_csv(path / file_names[1])
    return application_details, application_history