# Project Baikal

## Main goal
Provide a Keras/TensorFlow like API to develop complex machine learning pipelines. The API is mainly for pipelines that use algorithms other than neural networks (that's what TensorFlow, pytorch, Caffe, etc is for. That said, a neural network could be a component in the pipeline).

## Design guidelines
- API should be something like:

```python
# x1, x2, x1 ... y are data placeholders with pointers to the actual results,
# the actual results will be realized once the graph is run (feed_forward)
# (conceptually, kinda similar to a Tensor in Tensorflow)
x1 = ArrayNode(name='input1')
x2 = ArrayNode(name='input2')

z1 = SomePreprocessor(...)(x1)
z2 = SomePreprocessor(...)(x2)
c = Concatenate(...)([z1, z2])
y = SVM(..., name='y')(c)

# Define a model
model = Model([x1, x2], y)  # Could also be a sub-graph

# Train model a la Keras:
model.fit([x1_data, x2_data], y_data)

# or a la TensorFlow:
model.fit({'input1': x1_data,
           'input2': x2_data,
           'y': y_data})
             
# predict and feed_forward return concrete data (numpy arrays, pandas dataframes, etc)
y_pred = model.predict([x1_data, x2_data])

# Can also query specific outputs (and only give the necessary inputs)
outs = model.feed_forward({'input1': x1_data}, outputs=['z1'])
```

- Graph should be built dynamically (like TensorFlow)
    - There should be a default graph
    - Nodes should be added to the graph on instantiation
    - Edges should be added on calls to node `__call__` (e.g. `SomeNode(...)([n1, n2])`)
- Add model (graph) serialization and de-serialization
    - Something like `model.save(filename)` and `model.load(filename)`
    - Most likely pickle
        - Define `__getstate__` and `__setstate__` API
    - Consider using joblib, yaml?
- Should be able to use models as components of a bigger graph
    - Models should be callable on `ArrayNode`s
- A cache to avoid repeating the computations if the input data hasn't changed
- Choose the type of output of classifiers (predicted label, predicted score)
    - Useful for model ensembling
- It should be possible to freeze models to avoid fitting (e.g. when we loaded a pretrained model)
- Should have a `set_params` and `get_params` for compatiblity with sklearn's GridSearch API.
- Use `networkx` for graph stuff. After having a working API, replace with a in-house module.
    - We don't need much from `networkx`. Just the primitive classes and the topological sort algorithm.

# Tests TODO list

## Implementation TODO
- [ ] Add a InstanceTracker metaclass to keep track of instances and issue names
- [ ] refactor ArrayNode and ProcessorNodeMixin as subclasses of a Node class
    - [ ] make issue_name a static method, or take it outside of the classes altogether
- [ ] Use itertools count to get and id?

## API test cases:

- [x] Can create an ArrayNode with a name
    - [x] Raises error if an ArrayNode is created with a name already used by another Node
    - [x] If name is not specified, a unique name should be generated
    
```python
x1 = ArrayNode(name='x1')
y1 = ArrayNode(name='y1')
```

- [x] Can extend a user-defined class by mixing with ProcessorNodeMixin.
    - The user-defined class must implement the sklearn API:
        - fit/predict ( + predict_proba/decision_function)
        - fit/transform
        - transform only (this is for nodes that do not require fit and just transform the data, e.g. Concatenate, Merge, Split, etc.)
        - get_params
        - set_params
    - Though not implemented explicitly, any class that is extended with ProcessorNodeMixin will be referred as ProcessorNode hereafter.

- [x] A ProcessorNode can be instantiated with a name.
    - [x] If name is not specified, a unique name should be generated
    - [x] Raises error if a Processor is created with a name already used by another Node
    
```python
class ProcessorNode(SomeClass, ProcessorNodeMixin):
    pass
    
p0 = ProcessorNode(name='p0')
p1 = ProcessorNode(name='p1')
```

- [ ] Can call a Processor component instance with two Arrays: X and y.
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

