# %%
import research.start  # noqa

# %%
import pandas as pd

from models.model import Model


df = pd.read_csv("data/x.csv")

model = Model()
