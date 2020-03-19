# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# authors: 
# Jalina Op de Beeck, Furkan Yasar, Michiel Milis,
# Jorg Vergracht, Mathijs Weegels, Tijs Van den Heuvel
# %%
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# %% [markdown]
#
# ## Loading the data

# %%


df = pd.read_csv("dat/AirQualityUCI.csv", sep="\t")
df = df.loc[df["PT08.S3(NOx)"] >= 0]
df = df.loc[df["PT08.S5(O3)"] >= 0]

df.columns = df.columns.str.strip()
feat = df[["PT08.S3(NOx)"]]
outc = df["PT08.S5(O3)"]

plt.scatter(feat["PT08.S3(NOx)"], outc,s=0.1)
plt.xlabel("PT08.S3(NOx)")
plt.ylabel("PT08.S5(O3)")
plt.title("Scatterplot data")
plt.show()

# %% [markdown]
#
# ## Splitting the data


# %%
feat_trai, feat_test, outc_trai, outc_test = train_test_split(
    feat, outc, test_size=0.33, random_state=42)

# %% [markdown]
#
# ## Fit and plot the models


# %%
degrees = [1,2,3,4,5,6,7,8,9,10]

feat_test = pd.DataFrame.sort_values(feat_test,by=["PT08.S3(NOx)"])

plt.scatter(feat,outc,s=0.1)

mse = []
mse_std = []

for i in range(len(degrees)):

    polynomial_features = PolynomialFeatures(degree=degrees[i],include_bias=False)
    linear_regression = LinearRegression()

    learner = Pipeline([("polynomial_features", polynomial_features),("linear_regression",linear_regression)])
    learner.fit(feat_trai,outc_trai)

    scores = cross_val_score(learner,feat_trai,outc_trai,scoring="neg_mean_squared_error",cv=10)
    mse.append(-scores.mean())
    mse_std.append(scores.std())

    outc_pred = learner.predict(feat_test)

    plt.plot(feat_test,outc_pred,label="Degree {}".format(degrees[i]))

plt.xlabel("PT08.S3(NOx)")
plt.ylabel("PT08.S5(O3)")
plt.title("Polynomial models")
plt.legend(fontsize=8 ,ncol=2)
plt.show()

# %% [markdown]
#
# ## Optimal degree
#
# %%
for i in range(len(degrees)):
    message = "Degree {}: MSE = {:.2f}(+/- {:.2f})".format(degrees[i],mse[i],mse_std[i])
    print(message)

mse_min = min(mse)
index = mse.index(mse_min)
ideal_degree = index + 1
message = "The optimal polynomial degree is {}.".format(ideal_degree)

print(message)




