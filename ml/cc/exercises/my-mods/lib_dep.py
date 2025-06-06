#Allows python to run bash commands
import subprocess

#Install required libraries
bash_command = """
pip install keras~=3.8.0 \
  matplotlib~=3.10.0 \
  numpy~=2.0.0 \
  pandas~=2.2.0 \
  tensorflow~=2.18.0
  pip install -U kaleido"""

subprocess.run(bash_command, shell = True)

print("\n\nAll requirements sucessfully installed.")

#Loading dependencies

#general
import io

#data
import numpy as np
import pandas as pd

#machine learning
import keras

#data visualization
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
