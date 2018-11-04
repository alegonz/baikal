Test cases:

- Can create an ArrayNode with a name
    - Raises error if an ArrayNode is created with a name already used by another Node
    - If name is not specified, a unique name should be generated
    
```python
x1 = ArrayNode(name='x1')
y1 = ArrayNode(name='y1')
```

- Can extend a user-defined class by mixing with ProcessorNodeMixin.
    - The user-defined class must implement the sklearn API:
        - fit/predict ( + predict_proba/decision_function)
        - fit/transform
        - transform only (this is for nodes that do not require fit and just transform the data, e.g. Concatenate, Merge, Split, etc.)
        - get_params
        - set_params
    - Though not implemented explicitly, any class that is extended with ProcessorNodeMixin will be referred as ProcessorNode hereafter.

- A ProcessorNode can be instantiated with a name.
    - If name is not specified, a unique name should be generated
    - Raises error if a Processor is created with a name already used by another Node
    
```python
class ProcessorNode(SomeClass, ProcessorNodeMixin):
    pass
    
p0 = ProcessorNode(name='p0')
p1 = ProcessorNode(name='p1')
```

- Can call a Processor component instance with two Arrays: X and y.
    - The component must be defined by extending the original component with Processor mixin class
    - A component can only be called with ArrayPlaceholders as inputs
    - A component must return an ArrayNode
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
    - Models should be callable on `ArrayNode`s
- A cache to avoid repeating the computations if the input data hasn't changed
- Choose the type of output of classifiers (predicted label, predicted score)
    - Useful for model ensembling
- It should be possible to freeze models to avoid fitting (e.g. when we loaded a pretrained model)

# TODO
- Refactor Array as an instance of a metaclass that keeps tracks of its instances
