from functools import lru_cache
from typing import Union, List, Dict, Tuple

from baikal.core.data_placeholder import is_data_placeholder_list, DataPlaceholder
from baikal.core.digraph import DiGraph
from baikal.core.step import Step
from baikal.core.typing import ArrayLike
from baikal.core.utils import listify, find_duplicated_items


class Model(Step):
    def __init__(self, inputs, outputs, name=None):
        super(Model, self).__init__(name=name)

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

        self.inputs = inputs
        self.outputs = outputs
        self._graph = self._build_graph()
        self._all_steps_sorted = self._graph.topological_sort()  # Fail early if graph is acyclic
        self._data_placeholders = self._collect_data_placeholders(self._graph)

        self._get_required_steps = lru_cache(maxsize=128)(self._get_required_steps)
        self._get_required_steps(tuple(sorted(self.inputs)), tuple(sorted(self.outputs)))

        # TODO: Add a self.is_fitted flag?

    def _build_graph(self):
        # Model uses the DiGraph data structure to store and operate on its DataPlaceholder and Steps.
        graph = DiGraph()

        # Add nodes (steps)
        def collect_steps_from(output):
            parent_step = output.step
            graph.add_node(parent_step)
            for input in parent_step.inputs:
                collect_steps_from(input)

        for output in self.outputs:
            collect_steps_from(output)

        # Add edges (data)
        for step in graph:
            for input in step.inputs:
                graph.add_edge(input.step, step)

        # Check for any nodes (steps) with duplicated names
        duplicated_names = find_duplicated_items([step.name for step in graph])

        if duplicated_names:
            raise RuntimeError('A Model cannot contain steps with duplicated names!\n'
                               'Found the following duplicates:\n'
                               '{}'.format(duplicated_names))

        return graph

    @staticmethod
    def _collect_data_placeholders(graph):
        data_placeholders = set()
        for step in graph:
            for output in step.outputs:
                data_placeholders.add(output)
        return data_placeholders

    def _get_required_steps(self, inputs: Tuple[DataPlaceholder], outputs: Tuple[DataPlaceholder]) -> List[Step]:
        # inputs and outputs must be tuple (thus hashable) for lru_cache to work
        #
        # Backtrack from outputs until inputs to get the necessary steps. That is,
        # find the ancestors of the nodes that provide the specified outputs.
        # Raise an error if there is an ancestor whose input is not in the specified inputs.
        # We assume a DAG (guaranteed by success of topological_sort).

        all_required_steps = set()
        inputs_found = []

        # Depth-first search
        def backtrack(output):
            required_steps = set()

            if output in inputs:
                inputs_found.append(output)
                return required_steps

            parent_step = output.step
            if parent_step in all_required_steps:
                return required_steps

            required_steps = {parent_step}
            for input in parent_step.inputs:
                required_steps |= backtrack(input)
            return required_steps

        for output in outputs:
            all_required_steps |= backtrack(output)

        # Check for missing inputs
        missing_inputs = []
        for step in all_required_steps:
            if self._graph.in_degree(step) == 0:
                missing_inputs.extend(step.outputs)

        if missing_inputs:
            raise RuntimeError('The following inputs are required but were not specified:\n'
                               '{}'.format(','.join([input.name for input in missing_inputs])))

        # Check for any unused inputs
        for input in inputs:
            if input not in inputs_found:
                raise RuntimeError(
                    'Input {} was provided but it is not required to compute the specified outputs.'.format(input.name))

        return [step for step in self._all_steps_sorted if step in all_required_steps]

    def _normalize_data(self,
                        data: Union[ArrayLike, List[ArrayLike], Dict[DataPlaceholder, ArrayLike], Dict[str, ArrayLike]],
                        data_placeholders: List[DataPlaceholder],
                        expand_none=False) -> Dict[DataPlaceholder, ArrayLike]:
        if isinstance(data, dict):
            data_norm = {}
            for key, value in data.items():
                key = self._get_data_placeholder(key)
                data_norm[key] = value
        else:
            data = [None] * len(data_placeholders) if (data is None and expand_none) else data
            data_norm = dict(zip(data_placeholders, listify(data)))
        return data_norm

    def _get_data_placeholder(self, data_placeholder: Union[str, DataPlaceholder]) -> DataPlaceholder:
        # Steps are assumed to have unique names (guaranteed by success of _build_graph)
        # If the step names are unique, so are the data_placeholder names
        if isinstance(data_placeholder, str):
            for d in self._data_placeholders:
                if data_placeholder == d.name:
                    return d

        elif isinstance(data_placeholder, DataPlaceholder):
            if data_placeholder in self._data_placeholders:
                return data_placeholder

        raise ValueError('{} was not found in the model!'.format(data_placeholder))

    def fit(self, input_data, target_data=None):
        # TODO: add extra_targets keyword argument
        # TODO: Add **fit_params argument (like sklearn's Pipeline.fit)
        # TODO: Consider using joblib's Parallel and Memory classes to parallelize and cache computations
        # In graph parlance, the 'parallelizable' paths of a graph are called 'disjoint paths'
        # https://stackoverflow.com/questions/37633941/get-list-of-parallel-paths-in-a-directed-graph
        input_data = self._normalize_data(input_data, self.inputs)
        for input in self.inputs:
            if input not in input_data:
                raise ValueError('Missing input {}'.format(input))

        target_data = self._normalize_data(target_data, self.outputs, expand_none=True)
        for output in self.outputs:
            if output not in target_data:
                raise ValueError('Missing output {}'.format(output))

        steps = self._get_required_steps(tuple(sorted(input_data)), tuple(sorted(target_data)))

        results_cache = dict()  # keys: DataPlaceholder instances, values: actual data (e.g. numpy arrays)
        results_cache.update(input_data)

        for step in steps:
            # 1) Fit phase
            Xs = [results_cache[i] for i in step.inputs]
            if hasattr(step, 'fit'):
                # Filtering out None target_data allow us to define fit methods without y=None.
                ys = [target_data[o] for o in step.outputs if o in target_data and target_data[o] is not None]
                step.fit(*Xs, *ys)

            # 2) predict/transform phase
            self._compute_step(step, Xs, results_cache)

    def predict(self, input_data, outputs=None):
        results_cache = dict()  # keys: DataPlaceholder instances, values: actual data (e.g. numpy arrays)

        # Normalize inputs and outputs
        input_data = self._normalize_data(input_data, self.inputs)

        if outputs is None:
            outputs = self.outputs
        else:
            outputs = listify(outputs)
            outputs = [self._get_data_placeholder(output) for output in outputs]
            if len(set(outputs)) != len(outputs):
                raise ValueError('outputs must be unique.')

        steps = self._get_required_steps(tuple(sorted(input_data)), tuple(sorted(outputs)))

        # Compute
        results_cache.update(input_data)

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
        if hasattr(step, 'predict'):
            output_data = step.predict(*Xs)
        elif hasattr(step, 'transform'):
            output_data = step.transform(*Xs)
        else:
            raise TypeError('{} must implement either predict or transform!'.format(step.name))

        cache.update(zip(step.outputs, listify(output_data)))

    # TODO: Override __call__ method
    # query: inputs (outputs) can be a dictionary keyed by DataPlaceholder instances or
    # their names, with array values. We need input normalization for this.
    # Also, check that all of requested output keys exist in the Model graph.

    # For testing purposes
    @property
    def _steps_cache_info(self):
        return self._get_required_steps.cache_info()
