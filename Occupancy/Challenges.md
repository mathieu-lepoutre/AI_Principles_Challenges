
# Artificial Intelligence Principles - Challenges

32318/1700/1920/1/22
D'Haese David

---

## Occupancy

The goals of your first assignment is to reproduce a part of the work published by Candanedo and Feldheim (2016)<sup>1</sup> (see `Occupancy.ipynb`). The authors have tried to predict the presence of a human in a room based on the reading of 4 sensors measuring light, temperature, humidity and CO<sub>2</sub> concentration.
​
> __NOTE__: On this first assignment, you will be getting all a Jupyter Notebook for free. Soon you learn how to come up with a report like this one *from scratch*. For now, there will be a number of small challenges to complete, instead.
​
Start by running this document, either all cells at once or cell-by-cell. If you get some errors, try to see if installing the necessary modules helps. See details in the course text.

### Challenge I.01

Build a dependency tree similar as the one shown in `Theory>Importing Data Science Modules` to find dependencies among the modules `numpy`, `sklearn`, `pandas`, `plotly` and `matplotlib`.

```txt
Your answer here
```

### Challenge I.02

When running the Occupancy Notebook in VS Code, you have more control. Whenever you run a code cell using the links immediately above the cell, you should see the result appearing in an interactive window. In the top buttonbar of the interactive window, find the button to see all the active variables and press it. What type does this pane show for the variable `GaussianNB`?

```txt
Your answer here
```

### Challenge I.03

In the Occupancy Notebook, you see the following variables being created: `feat_trai, feat_test, outc_trai, outc_test`. What is the risk of features being separated from the outcome and of test instances being separated from the training instances? Explain in your own words:

```txt
Your answer here
```

### Challenge I.04

If all goes well, you do not need to inspect the basic data science modules such as `sklearn` or `numpy`. But sometimes things go wrong or you need to understand how things work 'under the hood'. Let us do some digging. Examine the `train_test_split` function in the `_split.py` file and answer the following questions:

- This function can split the set of instances into a training and test set, but what can it optionally also do (and explain what it means)?

```text
S..., this means ...
```

- Explain in your own words the impact of the `shuffle` parameter?

```text
Your answer here
```

- If you dig deeper in the code base, you will discover that during the split operation python actually performs a shuffle followed by an indexing operation. What is the name of the method in the `numpy.random` module that is responsible for the shuffling.

```text
Your answer here
```

### Challenge I.05

Let us check whether a lot of CO<sub>2</sub> measurements fall outside the accepted threshold of 1000 ppm (see the Wikipedia page on Indoor air quality<sup>2</sup>). What is the mean ± standard deviation of the CO<sub>2</sub> measurements according to the `describe` function:

```text
... ± ...
```

### Challenge I.06

Some learners are able make probabilistic predictions of categorical variables and this is also the case for the Naïve Bayes learner. Execute the command `learner.predict_proba(feat_test)` and describe how it's output differs from that of the `learner.predict` method. Take the first number in the output list and write a full sentence to explain what the number(s) actually mean(s).

```text
Your answer here
```

### Challenge I.07

Using our trained learner, predict whether there is a person in the room (both deterministic and probabilistic) given a light intensity of 500 lux and a CO<sub>2</sub>-reading of 615 ppm.

```text
Your answer here
```

### Challenge I.08

Prove that the accuracy of 97.35% can be calculated manually as the percentage of correctly predicted outcomes.

```python
Your answer here
```

### Challenge I.09

Prove that the accuracy of 97.35% can be calculated from the confusion matrix.

```python
Your answer here
```

### References

1. Candanedo, L. M., & Feldheim, V. (2016). Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. Energy and Buildings, 112, 28-39.
2. Wikipedia contributors. (2020, January 23). Indoor air quality. In Wikipedia, The Free Encyclopedia. Retrieved 17:25, January 26, 2020, from https://en.wikipedia.org/w/index.php?title=Indoor_air_quality&oldid=937158993.
