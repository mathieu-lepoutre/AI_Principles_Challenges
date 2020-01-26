# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
#  # AI Principles Challenges
# 
#  ## Challenge I: Occupancy
# 
#  The goals of your first assignment is to reproduce a part of the work published by Candanedo and Feldheim (2016)<sup>1</sup>. The authors have tried to predict the presence of a human in a room based on the reading of 4 sensors measuring light, temperature, humidity and CO<sub>2</sub> concentration.

# %%
import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from time import perf_counter
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, plot_roc_curve
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## Data
# 
# The data consist of the file `occupancy.txt` in the `dat` subfolder and was originally collected from the [UCI repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00357/) and then adjusted as follows:
# 
# 1. Merge test data and training data into single file
# 2. Rename column headers for the sake of clarity and consistency:
#   - `date` &rarr; `Time`
#   - `HumidityRatio` &rarr; `Humidity_Ratio`
# 3. Convert separator comma's to tabs
# 
# Of the features of the data set, only the reading from the light and CO<sub>2</sub> sensors will be retained.
# %% [markdown]
# ## Algorithm Training
# 
# The retained data set is split into 2 parts, one for training (&frac23;) and one for testing (&frac13;)

# %%
t0 = perf_counter()
data_df = pd.read_csv("dat/occupancy.txt", sep="\t")
feat = data_df[["Light", "CO2"]]
outc = data_df.Occupancy
feat_trai, feat_test, outc_trai, outc_test = train_test_split(
    feat, outc, test_size=0.33, random_state=42)

learner = GaussianNB()
learner.fit(feat_trai, outc_trai)

t1 = perf_counter()

print(f"Algorithm trained in {t1-t0:.3f} seconds.")

# %% [markdown]
# ## Algorithm Testing
# 
# We will now make  predictions on test data and calculate the performance.

# %%
outc_pred = learner.predict(feat_test)
outc_pred_proba = learner.predict_proba(feat_test)
performance = accuracy_score(outc_pred, outc_test)

print(f"Performance of the classifier when using Light and CO2 features:\n{performance:.2%}")

# %% [markdown]
# ## Visualization of Prediction
# 
# As we are focusing here on two features only, we can plot the the decision boundary overlaid by the predictions:

# %%
# Plot the decision boundary
x_min = feat.Light.min()
x_max = 900 # feat.Light.max()
y_min = feat.CO2.min()
y_max = feat.CO2.max()
n = 100

x_ax = np.linspace(x_min, x_max, n)
y_ax = np.linspace(y_min, y_max, n)
xx, yy = np.meshgrid(x_ax, y_ax)
z = learner.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.pcolormesh(xx, yy, z, cmap=plt.cm.coolwarm)

# Overlay with test points
plt.scatter(feat_test.Light.loc[outc_test == 1], feat_test.CO2[outc_test == 1],
    marker=".", color="r", label="present")
plt.scatter(feat_test.Light.loc[outc_test == 0], feat_test.CO2[outc_test == 0],
    marker=".", color="b", label="absent")

plt.legend()
plt.xlabel("Light")
plt.ylabel("CO2")

plt.show()

# %% [markdown]
# ## Algorithm Performance
# 
# The ROC curve below demonstrates the classification success independent of a threshold.

# %%
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
dummy = plot_roc_curve(learner, feat_test, outc_test, ax=ax )
plt.show()

# %% [markdown]
# ## Conclusion
# 
# In this report, we have tried to predict the presence of a human in a room based on the readings of two sensors only, light and CO<sub>2</sub>. With an overall performance of 97.35% on a randomly picked test data set (&frac13; of total size) and an AUC of 99%, we can say that, based on the presented data, on average only one in every 35 cases will be wrongly classified.
# %% [markdown]
# ## References
# 
#  1. Candanedo, L. M., & Feldheim, V. (2016). Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. Energy and Buildings, 112, 28-39.
