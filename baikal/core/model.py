from typing import Union, List, Dict, Sequence

from baikal.core.data_placeholder import is_data_placeholder_list, DataPlaceholder
from baikal.core.digraph import DiGraph
from baikal.core.step import Step
from baikal.core.typing import ArrayLike
from baikal.core.utils import listify, find_duplicated_items, SimpleCache


class Model(Step):
    def __init__(self, inputs, outputs, name=None, trainable=True):
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
        self._graph, self._data_placeholders, self._steps = self._build_graph()
        self._all_steps_sorted = self._graph.topological_sort()  # Fail early if graph is acyclic

        self._steps_cache = SimpleCache()
        self._get_required_steps(self._internal_inputs, self._internal_outputs)

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

        for output in self._internal_outputs:
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

        # Collect data placeholders
        data_placeholders = {}
        for step in graph:
            for output in step.outputs:
                data_placeholders[output.name] = output

        # Collect steps
        steps = {step.name: step for step in graph}

        return graph, data_placeholders, steps

    def _get_required_steps(self, inputs: Sequence[DataPlaceholder], outputs: Sequence[DataPlaceholder]) -> List[Step]:
        # Backtrack from outputs until inputs to get the necessary steps. That is,
        # find the ancestors of the nodes that provide the specified outputs.
        # Raise an error if there is an ancestor whose input is not in the specified inputs.
        # We assume a DAG (guaranteed by success of topological_sort).
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
                raise RuntimeError(
                    'Input {} was provided but it is not required to compute the specified outputs.'.format(input.name))

        required_steps = [step for step in self._all_steps_sorted if step in required_steps]
        self._steps_cache[cache_key] = required_steps

        return required_steps

    def _normalize_data(self,
                        data: Union[ArrayLike, List[ArrayLike], Dict[DataPlaceholder, ArrayLike], Dict[str, ArrayLike]],
                        data_placeholders: List[DataPlaceholder],
                        expand_none=False) -> Dict[DataPlaceholder, ArrayLike]:
        if isinstance(data, dict):
            data_norm = {}
            for key, value in data.items():
                key = self.get_data_placeholder(key.name if isinstance(key, DataPlaceholder) else key)
                data_norm[key] = value
        else:
            data = [None] * len(data_placeholders) if (data is None and expand_none) else data
            data_norm = dict(zip(data_placeholders, listify(data)))
        return data_norm

    def get_step(self, name: str) -> Step:
        # Steps are assumed to have unique names (guaranteed by success of _build_graph)
        if name in self._steps.keys():
            return self._steps[name]
        raise ValueError('{} was not found in the model!'.format(name))

    def get_data_placeholder(self, name: str) -> DataPlaceholder:
        # If the step names are unique, so are the data_placeholder names
        if name in self._data_placeholders.keys():
            return self._data_placeholders[name]
        raise ValueError('{} was not found in the model!'.format(name))

    def fit(self, input_data, target_data=None):
        # TODO: add extra_targets keyword argument
        # TODO: Add **fit_params argument (like sklearn's Pipeline.fit)
        # TODO: Consider using joblib's Parallel and Memory classes to parallelize and cache computations
        # In graph parlance, the 'parallelizable' paths of a graph are called 'disjoint paths'
        # https://stackoverflow.com/questions/37633941/get-list-of-parallel-paths-in-a-directed-graph
        input_data = self._normalize_data(input_data, self._internal_inputs)
        for input in self._internal_inputs:
            if input not in input_data:
                raise ValueError('Missing input {}'.format(input))

        target_data = self._normalize_data(target_data, self._internal_outputs, expand_none=True)
        for output in self._internal_outputs:
            if output not in target_data:
                raise ValueError('Missing output {}'.format(output))

        steps = self._get_required_steps(input_data, target_data)

        results_cache = dict()  # keys: DataPlaceholder instances, values: actual data (e.g. numpy arrays)
        results_cache.update(input_data)

        for step in steps:
            # 1) Fit phase
            Xs = [results_cache[i] for i in step.inputs]
            if hasattr(step, 'fit') and step.trainable:
                # Filtering out None target_data allow us to define fit methods without y=None.
                ys = [target_data[o] for o in step.outputs if o in target_data and target_data[o] is not None]
                # TODO: Add a try/except to catch missing target data errors (e.g. when forgot ensemble targets)
                step.fit(*Xs, *ys)

            # 2) predict/transform phase
            self._compute_step(step, Xs, results_cache)

    def predict(self, input_data, outputs=None):
        results_cache = dict()  # keys: DataPlaceholder instances, values: actual data (e.g. numpy arrays)

        # Normalize inputs and outputs
        input_data = self._normalize_data(input_data, self._internal_inputs)

        if outputs is None:
            outputs = self._internal_outputs
        else:
            outputs = listify(outputs)
            outputs = [self.get_data_placeholder(output) for output in outputs]
            if len(set(outputs)) != len(outputs):
                raise ValueError('outputs must be unique.')

        steps = self._get_required_steps(input_data, outputs)

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
