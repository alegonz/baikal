from collections import defaultdict
from typing import Union, List, Dict, Sequence, Optional, Iterable

from baikal._core.data_placeholder import is_data_placeholder_list, DataPlaceholder
from baikal._core.digraph import DiGraph
from baikal._core.step import Step, InputStep
from baikal._core.typing import ArrayLike
from baikal._core.utils import find_duplicated_items, listify, safezip2, SimpleCache


# Just to avoid function signatures painful to the eye
strs = Union[str, List[str]]
DataPlaceHolders = Union[DataPlaceholder, List[DataPlaceholder]]
ArrayLikes = Union[ArrayLike, List[ArrayLike]]
DataDict = Dict[Union[DataPlaceholder, str], ArrayLike]


class Model(Step):
    """A Model is a network (more precisely, a directed acyclic graph) of Steps,
    and it is defined from the input/output specification of the pipeline.
    Models have fit and predict routines that, together with graph-based engine,
    allow the automatic (feed-forward) computation of each of the pipeline steps
    when fed with data.

    Parameters
    ----------
    inputs
        Inputs to the model.

    outputs
        Outputs of the model.

    name
        Name of the model (optional). If no name is passed, a name will be
        automatically generated.

    trainable
        Whether the model is trainable (True) or not (False). Setting
        `trainable=False` freezes the model. This flag is only meaningful when
        using the model as a step in a bigger model.

    Attributes
    ----------
    graph
        The graph associated to the model built from the input/output
        specification.

    Methods
    -------
    fit
        Trains the model on the given input and target data.

    predict
        Generates predictions from the input data. It can also be used to
        query intermediate outputs.

    get_step
        Get a step (graph node) in the model by name.

    get_data_placeholder
        Get a data placeholder (graph half-edge) in the model by name.

    get_params
        Get parameters of the model.

    set_params
        Set the parameters of the model.
    """

    def __init__(self,
                 inputs: DataPlaceHolders,
                 outputs: DataPlaceHolders,
                 name: Optional[str] = None,
                 trainable: bool = True):
        super(Model, self).__init__(name=name, trainable=trainable)

        inputs = listify(inputs)
        if not is_data_placeholder_list(inputs):
            raise ValueError('inputs must be of type DataPlaceholder.')
        if len(set(inputs)) != len(inputs):
            raise ValueError('inputs must be unique.')

        outputs = listify(outputs)
        if not is_data_placeholder_list(outputs):
            raise ValueError('outputs must be of type DataPlaceholder.')
        if len(set(outputs)) != len(outputs):
            raise ValueError('outputs must be unique.')

        self.n_outputs = len(outputs)
        self._internal_inputs = inputs
        self._internal_outputs = outputs
        self._graph = None
        self._data_placeholders = None
        self._steps = None
        self._all_steps_sorted = None
        self._steps_cache = None
        self._build()

    def _build(self):
        # Model uses the DiGraph data structure to store and operate on its DataPlaceholder and Steps.
        self._graph = build_graph_from_outputs(self._internal_outputs)

        # Collect data placeholders
        self._data_placeholders = {}
        for step in self._graph:
            for output in step.outputs:
                self._data_placeholders[output.name] = output

        # Collect steps
        self._steps = {step.name: step for step in self._graph}

        self._all_steps_sorted = self._graph.topological_sort()  # Fail early if graph is acyclic
        self._steps_cache = SimpleCache()
        self._get_required_steps(self._internal_inputs, self._internal_outputs)

    def _get_required_steps(self,
                            inputs: Sequence[DataPlaceholder],
                            outputs: Sequence[DataPlaceholder]) -> List[Step]:
        """Backtrack from outputs until inputs to get the necessary steps.
        That is, find the ancestors of the nodes that provide the specified
        outputs. Raise an error if there is an ancestor whose input is not in
        the specified inputs. We assume a DAG (guaranteed by success of
        topological_sort).
        """
        cache_key = (tuple(sorted(inputs)), tuple(sorted(outputs)))
        if cache_key in self._steps_cache:
            return self._steps_cache[cache_key]

        required_steps = set()
        inputs_found = []

        # Depth-first search
        def backtrack(output):
            steps_required_by_output = set()

            if output in inputs:
                inputs_found.append(output)
                return steps_required_by_output

            parent_step = output.step
            if parent_step in required_steps:
                return steps_required_by_output

            steps_required_by_output = {parent_step}
            for input in parent_step.inputs:
                steps_required_by_output |= backtrack(input)
            return steps_required_by_output

        for output in outputs:
            required_steps |= backtrack(output)

        # Check for missing inputs
        missing_inputs = []
        for step in required_steps:
            if self._graph.in_degree(step) == 0:
                missing_inputs.extend(step.outputs)

        if missing_inputs:
            raise RuntimeError('The following inputs are required but were not specified:\n'
                               '{}'.format(','.join([input.name for input in missing_inputs])))

        # Check for any unused inputs
        for input in inputs:
            if input not in inputs_found:
                raise RuntimeError('Input {} was provided but it is not required '
                                   'to compute the specified outputs.'.format(input.name))

        required_steps = [step for step in self._all_steps_sorted if step in required_steps]
        self._steps_cache[cache_key] = required_steps

        return required_steps

    def _normalize_data(self,
                        data: Union[ArrayLikes, DataDict],
                        data_placeholders: List[DataPlaceholder],
                        expand_none=False) -> Dict[DataPlaceholder, ArrayLike]:
        if isinstance(data, dict):
            return self._normalize_dict(data)
        else:
            return self._normalize_list(data, data_placeholders, expand_none)

    def _normalize_dict(self, data: DataDict) -> Dict[DataPlaceholder, ArrayLike]:
        data_norm = {}
        for key, value in data.items():
            key = self.get_data_placeholder(key.name if isinstance(key, DataPlaceholder) else key)
            data_norm[key] = value
        return data_norm

    @staticmethod
    def _normalize_list(data: ArrayLikes,
                        data_placeholders: List[DataPlaceholder],
                        expand_none) -> Dict[DataPlaceholder, ArrayLike]:
        if data is None and expand_none:
            data = [None] * len(data_placeholders)
        else:
            data = listify(data)

        try:
            data_norm = dict(safezip2(data_placeholders, data))

        except ValueError as e:
            # TODO: Improve this message
            message = 'When passing inputs/outputs as a list or a single array, ' \
                      'the number of arrays must match the number of inputs/outputs ' \
                      'specified at instantiation. ' \
                      'Got {}, expected: {}.'.format(len(data), len(data_placeholders))
            raise ValueError(message) from e

        return data_norm

    def get_step(self, name: str) -> Step:
        """Get a step (graph node) in the model by name.

        Parameters
        ----------
        name
            Name of the step.

        Returns
        -------
        The step.
        """
        # Steps are assumed to have unique names (guaranteed by success of _build_graph)
        if name in self._steps.keys():
            return self._steps[name]
        raise ValueError('{} was not found in the model.'.format(name))

    def get_data_placeholder(self, name: str) -> DataPlaceholder:
        """Get a data placeholder (graph half-edge) in the model by name.

        Parameters
        ----------
        name
            Name of the data placeholder.

        Returns
        -------
        The data placeholder.
        """
        # If the step names are unique, so are the data_placeholder names
        if name in self._data_placeholders.keys():
            return self._data_placeholders[name]
        raise ValueError('{} was not found in the model.'.format(name))

    def fit(self,
            X: Union[ArrayLikes, DataDict],
            y: Optional[Union[ArrayLikes, DataDict]] = None,
            extra_targets: Optional[DataDict] = None,
            **fit_params):
        """Trains the model on the given input and target data.

        The model will automatically propagate the data through the pipeline and
        fit any internal steps that require training.

        Parameters
        ----------
        X
            Input data (independent variables). It can be either of:
                - A single array-like object (in the case of a single input)
                - A list of array-like objects (in the case of multiple inputs)
                - A dictionary mapping DataPlaceholders (or their names) to
                  array-like objects.
        y
            Target data (dependent variables) (optional). It can be either of:
                - None (in the case the single output is associated to a
                  non-trainable or unsupervised learning step)
                - A single array-like object (in the case of a single output)
                - A list of the above (in the case of multiple outputs)
                - A dictionary mapping DataPlaceholders (or their names) to
                  array-like objects or None. You can also include target data
                  required by intermediate steps not specified in the model outputs.
        extra_targets
            Target data required by intermediate steps not specified in the
            model outputs. If specified, it must be a dictionary mapping
            DataPlaceholders (or their names) to array-like objects or None.

            While contents of `extra_targets` can be included in the contents of
            `y`, this separate argument exists to pass target data to nested models.

        fit_params
            Parameters passed to the fit method of each model step, where each
            parameter name has the form ``<step-name>__<parameter-name>``.

        Returns
        -------

        """
        # TODO: Add better error message to know which step failed in case of any error
        # TODO: Consider using joblib's Parallel and Memory classes to parallelize and cache computations
        # In graph parlance, the 'parallelizable' paths of a graph are called 'disjoint paths'
        # https://stackoverflow.com/questions/37633941/get-list-of-parallel-paths-in-a-directed-graph

        # input/output normalization
        X = self._normalize_data(X, self._internal_inputs)
        for input in self._internal_inputs:
            if input not in X:
                raise ValueError('Missing input {}.'.format(input))

        y = self._normalize_data(y, self._internal_outputs, expand_none=True)
        for output in self._internal_outputs:
            if output not in y:
                raise ValueError('Missing output {}.'.format(output))

        if extra_targets is not None:
            y.update(self._normalize_dict(extra_targets))

        # Get steps and their fit_params
        steps = self._get_required_steps(X, y)
        fit_params_steps = defaultdict(dict)
        for param_key, param_value in fit_params.items():
            # TODO: Add check for __. Add error message if step was not found
            step_name, _, param_name = param_key.partition('__')
            step = self.get_step(step_name)
            fit_params_steps[step][param_name] = param_value

        # Intermediate results are stored here
        # keys: DataPlaceholder instances, values: actual data (e.g. numpy arrays)
        results_cache = dict()
        results_cache.update(X)

        for step in steps:
            Xs = [results_cache[i] for i in step.inputs]

            # TODO: Use fit_transform if step has it
            # 1) Fit phase
            if hasattr(step, 'fit') and step.trainable:
                # Filtering out None y's allow us to define fit methods without y=None.
                ys = [y[o] for o in step.outputs
                      if o in y and y[o] is not None]

                fit_params = fit_params_steps.get(step, {})

                # TODO: Add a try/except to catch missing output data errors (e.g. when forgot ensemble outputs)
                step.fit(*Xs, *ys, **fit_params)

            # 2) predict/transform phase
            self._compute_step(step, Xs, results_cache)

        return self

    def predict(self,
                X: Union[ArrayLikes, DataDict],
                outputs: Optional[Union[strs, DataPlaceHolders]] = None) -> ArrayLikes:
        """

        **Models are query-able**. That is, you can request other outputs other
        than those specified at model instantiation. This allows querying
        intermediate outputs and ease debugging.

        Parameters
        ----------
        X
            Input data. It follows the same format as in the fit function.

        outputs
            Required outputs (optional). You can specify any final or intermediate
            output by passing the name of its associated data placeholder. If
            not specified, it will return the outputs specified at instantiation.

        Returns
        -------
        The computed outputs.
        """
        # Intermediate results are stored here
        # keys: DataPlaceholder instances, values: actual data (e.g. numpy arrays)
        results_cache = dict()

        # Normalize inputs and outputs
        X = self._normalize_data(X, self._internal_inputs)

        if outputs is None:
            outputs = self._internal_outputs
        else:
            outputs = listify(outputs)
            if len(set(outputs)) != len(outputs):
                raise ValueError('outputs must be unique.')
            outputs = [self.get_data_placeholder(output) for output in outputs]

        steps = self._get_required_steps(X, outputs)

        # Compute
        results_cache.update(X)

        for step in steps:
            Xs = [results_cache[i] for i in step.inputs]
            self._compute_step(step, Xs, results_cache)

        output_data = [results_cache[o] for o in outputs]
        if len(output_data) == 1:
            return output_data[0]
        else:
            return output_data

    @staticmethod
    def _compute_step(step, Xs, cache):
        # TODO: Raise warning if computed output is already in cache.
        # This happens when recomputing a step that had a subset of its outputs already passed in the inputs.
        # TODO: Some regressors have extra options in their predict method, and they return a tuple of arrays.
        # https://scikit-learn.org/stable/glossary.html#term-predict
        output_data = step.compute(*Xs)
        output_data = listify(output_data)

        try:
            cache.update(safezip2(step.outputs, output_data))
        except ValueError as e:
            message = 'The number of output data elements ({}) does not match ' \
                      'the number of {} outputs ({}).'.format(len(output_data), step.name, len(step.outputs))
            raise RuntimeError(message) from e

    def get_params(self, deep=True):
        """Get the parameters of the model.

        Parameters
        ----------
        deep
            Get the parameters of any nested models.

        Returns
        -------
        params
            Parameter names mapped to their values.
        """
        # InputSteps are excluded
        params = {}
        for step in self._steps.values():
            if isinstance(step, InputStep):
                continue
            params[step.name] = step
            if hasattr(step, 'get_params'):
                for param_name, value in step.get_params(deep).items():
                    params['{}__{}'.format(step.name, param_name)] = value
        return params

    def set_params(self, **params):
        """Set the parameters of the model.

        Parameters
        ----------
        params
            Dictionary mapping parameter names to their values. Valid parameter
            of the form ``<step-name>__<parameter-name>``). Entire steps can
            be replaced with ``<step-name>`` keys.

            Valid parameter keys can be listed with get_params().

        Returns
        -------
        self
        """
        # ----- 1. Replace steps
        for key in list(params.keys()):
            if key in self._steps:
                self._replace_step(key, params.pop(key))

        # ----- 2. Replace each step params
        # Collect params by step
        step_params = defaultdict(dict)
        for key, value in params.items():
            step_name, _, param_name = key.partition('__')
            step_params[step_name][param_name] = value

        # Set params for each step
        for step_name, params in step_params.items():
            step = self.get_step(step_name)
            step.set_params(**params)

        return self

    def _replace_step(self, step_key, new_step):
        # Transfer connectivity configuration from old step
        # to new step and replace old with new
        transfer_attrs = ['name', 'trainable', 'inputs', 'outputs']
        old_step = self._steps[step_key]
        for attr in transfer_attrs:
            setattr(new_step, attr, getattr(old_step, attr))

        # Update outputs of old step to point to the new step
        for output in old_step.outputs:
            output.step = new_step

        # Rebuild model
        self._build()

    @property
    def graph(self):
        return self._graph


def build_graph_from_outputs(outputs: Iterable[DataPlaceholder]) -> DiGraph:
    """Builds a graph by backtracking from a sets of outputs.

    It does so by backtracking recursively in depth-first fashion, jumping
    from outputs to steps in tandem until hitting a step with no inputs (an
    InputStep).

    Parameters
    ----------
    outputs
        Outputs (data placeholders) from where the backtrack to build the
        graph starts.

    Returns
    -------
    graph
        The built graph.
    """
    graph = DiGraph()

    # Add nodes (steps)
    def collect_steps_from(output):
        parent_step = output.step

        if parent_step in graph:
            return

        graph.add_node(parent_step)
        for input in parent_step.inputs:
            collect_steps_from(input)

    for output in outputs:
        collect_steps_from(output)

    # Add edges (data)
    for step in graph:
        for input in step.inputs:
            graph.add_edge(input.step, step, input)

    # Check for any nodes (steps) with duplicated names
    duplicated_names = find_duplicated_items([step.name for step in graph])

    if duplicated_names:
        raise RuntimeError('A graph cannot contain steps with duplicated names. '
                           'Found the following duplicates:\n'
                           '{}'.format(duplicated_names))

    return graph
