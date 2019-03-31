# Project Baikal

## Main goal
Provide a Keras/TensorFlow like API to develop complex machine learning pipelines. The API is mainly for pipelines that use algorithms other than neural networks (that's what TensorFlow, pytorch, Caffe, etc is for. That said, a neural network could be a component in the pipeline).

## Design guidelines
- API should be something like:

```python
# x1, x2, x1 ... y are DataPlaceholder objects with pointers to the actual results,
# the actual results will be realized once the graph is run (feed_forward)
# (conceptually, kinda similar to a Tensor in Tensorflow)
x1 = Input(name='input1')
x2 = Input(name='input2')

z1 = SomeStep(...)(x1)
z2 = SomeStep(...)(x2)
c = Concatenate(...)([z1, z2])
y = SVM(..., name='y')(c)

# Define a model
model = Model([x1, x2], y)  # Could also be a sub-graph

# Train model a la Keras:
model.fit([x1_data, x2_data], y_data)
             
# predict and feed_forward return concrete data (numpy arrays, pandas dataframes, etc)
y_pred = model.predict([x1_data, x2_data])

# Can also query specific outputs (and only give the necessary inputs)
outs = model.query({'input1': x1_data}, outputs=['z1'])
```

## Some desired features
- Add model (graph) serialization and de-serialization
    - Something like `model.save(filename)` and `model.load(filename)`
    - Most likely pickle
        - Define `__getstate__` and `__setstate__` API
    - Consider using joblib, yaml?
- Should be able to use models as components of a bigger graph
- Choose the type of output of classifiers (predicted label, predicted score)
    - Useful for model ensembling
- It should be possible to freeze models to avoid fitting (e.g. when we loaded a pretrained model)
- Should have a `set_params` and `get_params` for compatiblity with sklearn's GridSearch API.
- Provide a factory function to extend sklearn classes:
    - e.g.: `LogisticRegression = make_sklearn_step(sklearn.linear_model.logistic.LogisticRegression)`
- A cache to avoid repeating the computations if the input data hasn't changed

## Implementation ideas
- DiGraph is the internal representation of the DataPlaceholder and Steps that belong to a Model.
    - It is build on Model instantiation. 
- Represent graphs as dicts-of-sets (adjacency list)
    - This is similar to the data structure used in networkx (dicts-of-dicts)
    - Maybe only dicts (nodes) of sets (successors) are necessary
        - Do we also need a similar structure with the predecessors?
- The following abstractions should be made public to code:
    - Step
        - A node of the graph
        - Analogous to TensorFlow's Operation
        - Internally derived from a Node class
        - A step is callable on DataPlaceholder inputs
            - Provide a special 'target' argument for inputs required only at fit time. This allow us to treat label data as Inputs, transform them and connect them to e.g. classifier Steps. 
    - DataPlaceholder
        - An semi-edge of the graph
        - Analogous to TensorFlow's Tensor
        - Internally derived from a Edge class
    - Input
        - An special node of the graph that allows inputting data (arrays, dataframes, etc) from client code
            - At instantiation, internally it creates a special InputStep, and returns a DataPlaceholder object
        - Analogous to TensorFlow's Placeholder and Input in Keras
    - Model
        - A graph (of Step's that pass DataPlaceholder along each other) with defined inputs and outputs.
        - A graph with fit/predict API
        - Models must be defined (`__init__`) with DataPlaceholder inputs and outputs
        - Internally derived from a DiGraph class
        - Model fitting:
            - To fit a graph we need to:
                1. Filter the necessary steps depending on the provided inputs and outputs
                2. For each step (in topological order), we fit and then do transform/predict on the inputs, and store the result (node output) in the cache. This has to handle two cases:
                    1. The step is a transformer (Concatenate, Split, PCA, Scaler, etc). In this case we do not need target data, and we do fit_transform. The result of fit_transform becomes the node output.
                    2. The step is a Estimator (LogisticRegression, DecisionTree, etc). In this case we need target data, and we do fit and then predict. The result of predict becomes the node output.
                - While the Model API specifies only the final outputs and provides its target data via the Model.fit method, any intermediate trainable steps can take its required target data via a extra_targets argument in Model.fit.
        - Model persistence:
            - Should be able to save models to a file
            - Method 1:
                - Dumping:
                    - Persist the graph steps and their connections, but not the graph itself
                        - Perhaps in yaml/json or dot format. Preferably json (json is a python built-in module).
                    - Persist the steps coefficients/weights and their params
                    - This is similar to what Keras's `Model.save_model` does
                - Loading:
                    - Must regenerate the steps and add them to the client code's graph
                    - Must also load any coefficients/weights and params of each step (e.g. regression coefs, PCA components, etc)
                - Pros:
                    - More secure than pickle/joblib since does not execute any code.
                - Cons:
                    - Preferably models should be dumped/loaded like in scikit-learn:
                        - `dump(clf, 'filename.joblib'); clf = load('filename.joblib')`
                    - scikit-learn does not provide an API for persisting coefficients/weights independently.
            - Method 2:
                - Dumping:
                    - Dump entire Model to a pickle/joblib like you would do with a sklearn object.
                - Loading:
                    - Load entire Model to a pickle/joblib like you would do with a sklearn object.
                - Pros:
                    - Rather straightforward
                - Cons:
                    - What happens with the names of the loaded Steps and any already existing Steps?
                    - Have to make sure the Model internals are pickle-able.
                    - Inherent security issues of pickle/joblib.
            - Other considerations:
                - Models should freeze any fitted steps prior to saving to a file.
                    - Introduce an optional argument in `load_model`: `unfreeze_steps=False` 
                - If the user wants to re-train the whole model:
                    - Load the model with `unfreeze_steps=True`. This will unfreeze all the steps.
                - If the user wants to re-train some steps:
                    - Load the model and unfreeze the desired steps
        - A Model can be reused as a Step
            - Possibly in another graph? (in this case we would need to persist the graph with the Model)
            - Implement `__call__` method
                - Differs with Step `__call__`:
                    - Number of inputs should match with those used at `__init__`
                    - Outputs are already known from `__init__`
                    - Should compose Model graph into caller graph
        - Model visualization
            - Include a plot function that plots the model graph

### Compilation (i.e. call to `Model(...)`)
- [x] Do topological sort to get the order of execution of the steps.

### Feedforward
- [x] Find the required steps with a recursive predecessor search
    - Backtrace the predecesor nodes to find the inputs required to compute the specified outputs
        - Stop a backtrace path if the node's output is found in the provided inputs
- [x] Execute the required steps according to the topological sort
    - Use a results cache (just a dict)

## Tests TODO list

### API test cases:

- [x] Can create an Input with a name
    - [x] If name is not specified, a unique name should be generated
        - [x] Input (Node) naming format: `graph_name/node_name`
        - [x] DataPlaceholder (Node output, semi-edge) naming format: `graph_name/node_name/output_name` ?
    - [x] Creates another instance with an unique name if an Input is created with a name already used by another Input
    - [x] At instantiation:
        - [x] An InputStep is added to the default graph
        - [x] A DataPlaceholder object with the specified name is returned
    
```python
x1 = Input(name='x1')
x2 = Input(name='x2')
```

- [x] Can extend a user-defined class by mixing with Step.
    - The user-defined class must implement the sklearn API:
        - fit/predict ( + predict_proba/decision_function)
        - fit/transform
        - transform only (this is for nodes that do not require fit and just transform the data, e.g. Concatenate, Merge, Split, etc.)
        - get_params
        - set_params
    - Though not implemented explicitly, any class that is extended with Step will be referred as Step hereafter.

- [x] A Step can be instantiated with a name.
    - [x] If name is not specified, a unique name should be generated
    - [x] Creates another instance with a unique name if a Step is created with a name already used by another Step
    
```python
class SomeStep(Step, SomeSklearnClass):
    pass
    
p0 = SomeStep(name='p0')
p1 = SomeStep(name='p1')
```

- [x] Can call a Step component instance with (possibly several DataPlaceholder objects).
    - [x] A component must be defined by extending the original component with Step mixin class
    - [x] A component can only be called with DataPlaceholder objects as inputs
    - [x] A call to Step must return DataPlaceholder objects
```python
pred = SVC(...)(inputs=[x1, x2])
```

- [x] Can instantiate a Model with specified inputs and outputs (inputs and outputs are DataPlaceholder objects)
    - Both inputs and outputs are mandatory arguments
```python
model = Model(inputs=[x1, x2], outputs=[y1, y2])
```

- [x] Can fit the model a la Keras with lists of actual data (numpy arrays, pandas dataframes, etc)
    - [x] There are three cases for the target data:
        - All outputs require target data
            - `target_data` must match the number of outputs.
            - `target_data` must be a list if several outputs were specified. 
        - Some outputs require target data
            - Same as above, but we allow the elements of the outputs that do not require target data to be None.
        - None of the outputs require target data (for example, model only has transformers and/or unsupervised learning steps)
            - Same as above, but we also allow `target_data=None` even if multiple outputs were specified.
    - [x] Steps that do not implement a fit method should be skipped
    - [ ] Steps that are set to freeze (e.g. loaded a pretrained model) should be skipped
```python
model.fit([x1_data, x2_data], [y1_target_data, y2_target_data])
```

- [x] Can predict with the model a la Keras with lists of actual data (numpy arrays, pandas dataframes, etc)
    - If multiple outputs were specified, a list of actual result data is returned
```python
out = model.predict([x1_data, x2_data])
# out is actual result data (e.g. a numpy array or a list of numpy arrays) 
```

- [ ] Can query the model a la graphkit, with a dictionary of actual data (numpy arrays, pandas dataframes, etc)
```python
# input_data dictionary keys can be either DataPlaceholder objects or their name strings
# outputs can be a list of DataPlaceholder objects or their name strings
model.predict(input_data={'x1': ...}, outputs=[z1, y2])
# out = {'pred': ...}  # output dictionary keys should be name strings by default?
```

- [x] model.fit fails if any of the required inputs was not passed in
    - This includes target data for outputs in the case of supervised learning

- [x] model.predict fails if any of the required inputs is not in the provided inputs


### TODO 2019/03/29
- [x] `DataPlaceholder`
    - Rename to `DataPlaceholder` to avoid confusion with actual data, and to approximate the semantics of TensorFlow's placeholder
- [x] `Model`
    - [x] Fix huge bug in cache update in `fit`
    - [x] Test raises `NotFittedError` when predict is run before fit.
    - [x] Add check for step name uniqueness (and hence their outputs) when building
        - Raise error if duplicated names are found
    - [x] Extend `predict` method to handle input_data and request outputs other than those specified at instantiation
    - [x] Add steps cache
        - Need to make steps sortable     
    - [x] Implement `extra_targets` argument in `Model.fit`
        - Already possible with current implementation of `Model.fit(..., target_data)`
        - Test with a simple ensemble
    - [x] Add test for lru_cache with same inputs in different order
        - `_get_required_steps(sorted(tuple(inputs)), sorted(tuple(outputs)))`
        - lru_cache had to be replaced with homemade one to make Model easily picklable
    - [x] Ditched shape
        - After more careful thinking. Shape definition, check and inference is not necessary
        - At first I mistakenly thought shapes are needed to infer the number of a step outputs, but this is not the case.
            - The only useful non-user defined multiple output step is Split, and Split can guess the number of outputs from the splits argument.
            - Concatenate, Merge, etc, by definition, return exactly one output.
    - [x] Implement `Model.__call__`
        - No need to override super class
        - Can be achieved by distinguishing between inputs/outputs at `__init__` time (internal) and those at `__call__` 
    - [x] Extend graph building to handle `Model` steps
    - [x] Implement serialization
        - joblib and pickle should work just like that, no extra coding
- [x] `Step`
    - [x] Add `trainable=True` keyword argument to `__init__`
        - Add as an attribute so we can also do `step.trainable = False`
        - Condition the fit step in Model to this attribute

### TODO 2019/03/31
- [x] Check if `joblib.Parallel` allows nested calls
    - Apparently it does :)
    - Nested calls will happen when fitting/predicting with a big Model that contains inner (nested) Model steps.
- [ ] `Model`
    - [ ] Implement `get_params` and `set_params` API for compatibility with `GridSearchCV`
        - Also check how `Pipeline.fit` does this
    - [ ] Handle `**fit_params` argument (like sklearn's `Pipeline.fit`)
    - [ ] Add targets via inputs
        - Useful for implementing transformation pipelines for target data
        - Added via a optional argument in `Step.__call__`
            - e.g. `LogisticRegression()(inputs=x1, target_data=y_trans)  # y_trans is a DataPlaceholder object output from another Step`
            - When fitting, look for target_data in results cache instead of the provided `target_data`
    - [ ] Handle extra options in predict method
        - Some regressors have extra options in their predict method, and they return a tuple of arrays.
        - See: https://scikit-learn.org/stable/glossary.html#term-predict
        - Idea:
            - Add `**predict_params` argument to `Step.__init__`
                - This will add extra outputs.
                - This however, is class dependent
                - As far as I know, `predict_params` are boolean flags that choose whether to return extra arrays
                    - For example in `GaussianProcessRegressor` there are `return_std` and `return_cov` flags.
                    - However behavior is class dependent, for example, the `GaussianProcessRegressor` can only take either `return_std` or `return_cov`, not both at the same time.
    - [ ] Add parallelization with joblib (`Parallel` API)
        - [ ] Modify `_get_required_steps` to also return step depth
        - [ ] Create a step generator that feeds steps as soon as a processor becomes available
    - [ ] Add results caching with joblib (`Memory` API)

### TODO 2019/04/06
- [ ] Make steps shareable.
    - That is, they can be called multiple times on different inputs and connect them to different outputs.
    - Same concept as Keras' nodes.
- [ ] Make Model steps replaceable
    - This allows searching for estimators in e.g. GridSearchCV
- [ ] Add typing?
- [ ] Write documentation
- [ ] Extend API to handle more kind of step computations?
    - `predict_proba`, `score`, `sample`, etc.
    - This perhaps could be chosen at `__call__` time.
        - For example a `function` argument that takes the name of the function (or functions) used at predict time.
        - `y_pred, y_proba = LogisticRegression()(x1, function=['predict', 'predict_proba'])`
        
### Future?
- [ ] Make model topology settable
    - This could allow search on pipeline structures, not only on hyperparameters
    - How?

### Notes on Inputs/outputs normalization
- Behavior of methods
    - `__init__`
        - Arguments:
            - inputs: `Union[DataPlaceholder, List[DataPlaceholder]]`
            - outputs: `Union[DataPlaceholder, List[DataPlaceholder]]` (it is mandatory!)
        - Normalization:
            - inputs: `List[DataPlaceholder]`
            - outputs: `List[DataPlaceholder]`
            - Raises:
                - `ValueError` if:
                    - Any of the items is not of type `DataPlaceholder`
                    - Any of the items is duplicated (DataPlaceholder with duplicated names)
                        - Implement `__eq__` in DataPlaceholder class
        - Notes:
            - `__init__` should fail early leveraging the errors raised by `_build_graph` and `_get_required_steps` when:
                - The provided inputs and outputs involve steps with duplicated names (`_build_graph`)
                - The provided inputs and outputs lead to a cyclic graph (`_build_graph`)
                    - Something beautiful: The API itself makes building cyclic graph impossible
                        - (unless you do weird stuff like instantiating DataPlaceholder directly and messing with the internals of the steps and data instances).
                - Any of the provided inputs are actually not required to compute the specified outputs (`_get_required_steps`)
                - If any of the inputs required by the specified outputs were not provided (`_get_required_steps`)
            - This means that we force the user to define inputs and outputs properly

    - `fit`
        - Arguments:
            - input_data: `Union[ArrayLike, List[ArrayLike], Dict[str, ArrayLike], Dict[DataPlaceholder, ArrayLike]]`
            - target_data: `Optional[Union[ArrayLike, List[ArrayLike], Dict[str, ArrayLike], Dict[DataPlaceholder, ArrayLike]]]`
                - If List, for steps that do not require fit, items can be None
                - If Dict, for steps that do not require fit, keys can be omitted, or values can be None
        - Normalization:
            - input_data: `Dict[DataPlaceholder, ArrayLike]` (keys must match the inputs passed at instantiation)
            - target_data: `Dict[DataPlaceholder, ArrayLike]` (keys must match the outputs passed at instantiation)
                - If target_data is None, return `Dict[DataPlaceholder, None]` with keys equal to the outputs passed at instantiation
            - Raises:
                - `ValueError` when:
                    - if input_data was a passed as `ArrayLike`, or `List[ArrayLike]`
                        - the number of elements does not match the inputs passed at instantiation
                    - if target_data was a passed as `ArrayLike`, or `List[ArrayLike]`
                        - the number of elements does not match the outputs passed at instantiation
                    - if input_data was a passed as `Dict`
                        - the keys do not match the inputs passed at instantiation
                            - Must match in names and number
                        - the keys are a type other than `str` or `DataPlaceholder`
                    - if target_data was a passed as `Dict`
                        - the keys do not match the outputs passed at instantiation
                        - the keys are a type other than `str` or `DataPlaceholder`
        - Notes:
            - If normalization succeeds, `_get_required_steps` will succeed too.
                - inputs and outputs are the same as in `__init__`
    - `predict`
        - Arguments:
            - input_data: `Union[ArrayLike, List[ArrayLike], Dict[str, ArrayLike], Dict[DataPlaceholder, ArrayLike]]`
            - outputs: `Optional[Union[str, List[str], DataPlaceholder, List[DataPlaceholder]]]`
        - Normalization:
            - input_data: `Dict[DataPlaceholder, ArrayLike]`
            - outputs: `List[DataPlaceholder]`
                - If output is None, return the outputs defined at instantiation
            - Raises:
                - `ValueError` when:
                    - if input_data was a passed as `ArrayLike`, or `List[ArrayLike]`
                        - the number of elements does not match the inputs passed at instantiation
                    - if input_data was a passed as `Dict`
                        - the keys do not exist in the inputs passed at instantiation
                        - the keys are a type other than `str` or `DataPlaceholder`
                    - if outputs:
                        - Any of the items is not of type `DataPlaceholder` or `str`
                        - Any of the items is duplicated (DataPlaceholder with duplicated names)
        - Notes:
            - `predict` should fail early leveraging the errors raised by `_get_required_steps` when:
                - Any of the provided input_data are actually not required to compute the specified outputs
                - If any of the input_data required by the specified outputs were not provided
            - This means that we force the user to pass input_data and outputs properly
            
### Notes on `get_params` output
- `Model.get_params(deep=True)` should a have an output like below:
    - Should it include? These are not hyperparameters
        - `'inputs': list of data placeholders`
        - `'outputs': list of data placeholders`
        - `'name': Model name`
        - `'trainable': flag value`
- `Step` should also implement `get_params` and `set_params` ?

```
If deep=False:
{
# Maybe we should not return these
'inputs': list of data placeholders,
'outputs': list of data placeholders,
'name': Model name,
'trainable': flag value,
}

If deep=True:
{
# Maybe we should not return these
'inputs': list of data placeholders,
'outputs': list of data placeholders,
'name': Model name,
'trainable': flag value,

# Certainly return these
'step1': Step (or Model) instance,
'step2': Step (or Model) instance,
'step3': Step (or Model) instance,
'step1__p1': some value,
'step1__p2': some value,
...
'step2__p1': ...,
'step2__p2': ...,
...
# in case step3 is a (sub)Model instance:
'step3__step1': Step instance,
'step3__step2': Step instance,
'step3__step1__p1': some value,
'step3__step1__p2': some value,
'step3__step2__p1': some value,
'step3__step2__p2': some value,
...
}
```

- `Model.set_params(**kwargs)`:
    - Should ignore the following (raise a warning?):
        - `inputs`
        - `outputs`
        - `name`
        - `trainable`
    - Should accept a dictionary like the one produced by `get_params()`
- `Model.set_params(**Model.get_params())` should not change the parameters