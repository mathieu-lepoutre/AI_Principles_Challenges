
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
jupyter==1.0.0
  - ipykernel [required: Any, installed: 5.1.3]
  - ipywidgets [required: Any, installed: 7.5.1]
  - jupyter-console [required: Any, installed: 6.1.0]
  - nbconvert [required: Any, installed: 5.6.1]
  - notebook [required: Any, installed: 6.0.3]
  - qtconsole [required: Any, installed: 4.6.0]
matplotlib==3.1.2
  - numpy [required: >=1.11, installed: 1.18.1]
pandas==0.25.3
  - numpy [required: >=1.13.3, installed: 1.18.1]
sklearn==0.0
  - scikit-learn [required: Any, installed: 0.22.1]
    - numpy [required: >=1.11.0, installed: 1.18.1]
    - scipy [required: >=0.17.0, installed: 1.4.1]
      - numpy [required: >=1.13.3, installed: 1.18.1]
plotly==4.5.0
numpy==1.18.1
```

### Challenge I.02

When running the Occupancy Notebook in VS Code, you have more control. Whenever you run a code cell using the links immediately above the cell, you should see the result appearing in an interactive window. In the top buttonbar of the interactive window, find the button to see all the active variables and press it. What type does this pane show for the variable `GaussianNB`?

```txt
ABCMeta
```

### Challenge I.03

In the Occupancy Notebook, you see the following variables being created: `feat_trai, feat_test, outc_trai, outc_test`. What is the risk of features being separated from the outcome and of test instances being separated from the training instances? Explain in your own words:

```txt
If you start manipulating one of these variables, you have to remind yourself to immediately do the same operations on the other 3 variables. Otherwise the data might become corrupted and rendered useless.
```

### Challenge I.04

If all goes well, you do not need to inspect the basic data science modules such as `sklearn` or `numpy`. But sometimes things go wrong or you need to understand how things work 'under the hood'. Let us do some digging. Examine the `train_test_split` function in the `_split.py` file and answer the following questions:

- This function can split the set of instances into a training and test set, but what can it optionally also do (and explain what it means)?

```python
def train_test_split(*arrays, **options)
    Split arrays or matrices into random train and test subsets

    Quick utility that wraps input validation and next(ShuffleSplit().split(X, y)) and application to input data into a single call for splitting (__and optionally subsampling__) data in a oneliner.

    Read more in the :ref:User Guide <cross_validation>.
```

- Explain in your own words the impact of the `shuffle` parameter?

```txt
from *_split.py*: Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.

So setting shuffle to True will perform a true randomization, otherwise is is a simple split along the instance axis.
```

- If you dig deeper in the code base, you will discover that during the split operation python actually performs a shuffle followed by an indexing operation. What is the name of the method in the `numpy.random` module that is responsible for the shuffling.

```text
_split.py > train_test_split:

    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for "
                "shuffle=False")

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    else:
        if stratify is not None:
            CVClass = StratifiedShuffleSplit
        else:
            CVClass = ShuffleSplit

        cv = CVClass(test_size=n_test,
                     train_size=n_train,
                     random_state=random_state)

From the above we see some how the shuffling happens in the ShuffleSplit class. In there, we find def _iter_indices of class ShuffleSplit(BaseShuffleSplit) in _split.py: where the method permutation is being called on an object called rng:
        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            # random partition
            permutation = rng.permutation(n_samples)
            ind_test = permutation[:n_test]
            ind_train = permutation[n_test:(n_test + n_train)]
            yield ind_train, ind_test

In def check_random_state(seed) in validation.py, we find that in case the randomstate is provided as a number (42 in our case) a np.random.RandomState is being returned on which the permutation method is being called see [here](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.RandomState.permutation.html#numpy.random.RandomState.permutation).
```

### Challenge I.05

Let us check whether a lot of CO<sub>2</sub> measurements fall outside the accepted threshold of 1000 ppm (see the Wikipedia page on Indoor air quality<sup>2</sup>). What is the mean ± standard deviation of the CO<sub>2</sub> measurements according to the `describe` function:

```text
690 ± 311
```

### Challenge I.06

Some learners are able make probabilistic predictions of categorical variables and this is also the case for the Naïve Bayes learner. Execute the command `learner.predict_proba(feat_test)` and describe how it's output differs from that of the `learner.predict` method. Take the first number in the output list and write a full sentence to explain what the number(s) actually mean(s).

```text
The output from the proba function has two columns instead of one. For the first instance in feat_test, we see the output [1.09600034e-06, 9.99998904e-01]. The first number is the predicted probability of a human being absent from the room (Occupancy = 0) whereas the second number represents the probability that there is a human present (Occupancy = 1).
```

### Challenge I.07

Using our trained learner, predict whether there is a person in the room (both deterministic and probabilistic) given a light intensity of 500 lux and a CO<sub>2</sub>-reading of 615 ppm.

```text
It is predicted that there is a human present with a probability of >99.99%
```

### Challenge I.08

Prove that the accuracy of 97.35% can be calculated manually as the percentage of correctly predicted outcomes.

```python
> sum(outc_pred == outc_test)/len(outc_test)
0.9734708916728076
```

### Challenge I.09

Prove that the accuracy of 97.35% can be calculated from the confusion matrix.

```python
> (cm[0, 0]+cm[1, 1])/sum(sum(cm))

(cm[0, 0]+cm[1, 1])/sum(sum(cm))
0.9734708916728076
```

### References

1. Candanedo, L. M., & Feldheim, V. (2016). Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. Energy and Buildings, 112, 28-39.
2. Wikipedia contributors. (2020, January 23). Indoor air quality. In Wikipedia, The Free Encyclopedia. Retrieved 17:25, January 26, 2020, from https://en.wikipedia.org/w/index.php?title=Indoor_air_quality&oldid=937158993.
