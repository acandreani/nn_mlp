import pandas as pd
import numpy as np
import requests
import re
import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf


filename = "iris.data"


dataset = pd.read_csv(filename, header=None, names=['sepal_length','sepal_width','petal_length','petal_width','species'])
dataset.head()


seaborn.pairplot(dataset, height=2, diag_kind="kde")
plt.show()