Test cases:

- Can create an ArrayPlaceHolder with a name
```python
x1 = ArrayPlaceHolder(name='x1')
y1 = ArrayPlaceHolder(name='y1')
```

- Can create a SVC component instance

- Can call an SVC component instance with two ArrayPlaceHolders: X and y.
```python
y1_pred = SVC(...)(X=x1, y=y1)
```

- Can instantiate a Model with specified inputs and outputs
```python
model = Model(inputs=x1, outputs=y1_pred)
```

- Can fit the model with a dictionary of numpy arrays
```python
model.fit({x1: ..., y1: ...})
```

- Can call the model with a dictionary of numpy arrays for prediction and returns a dictionary with the outputs
```python
out = model.predict({x1: ...})
# out
# {y1_pred: ...}
```

- model.fit fails if any of the required inputs was not passed in the dictionary
    - Corollary: there is any learning component that 1) was not supplied y's AND 2) requires fit (and thus the y's).

- model.predict fails if any of the required inputs is not in the input dictionary

# Future work
- Add model (graph) serialization and de-serialization
    - Something like `model.dump(filename)` and `model.load(filename)`
    - Define `__getstate__` and `__setstate__` API
- Should be able to use models as components of a bigger graph
    - Models should be callable on `ArrayPlaceholder`s
- A cache to avoid repeating the computations if the input data hasn't changed
- Choose the type of output of classifiers (predicted label, predicted score)
  - Useful for model ensembling
