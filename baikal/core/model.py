from typing import Union, List, Dict

from baikal.core.data import is_data_list, Data
from baikal.core.digraph import DiGraph
from baikal.core.step import Step
from baikal.core.typing import ArrayLike
from baikal.core.utils import listify, find_duplicated_items


class Model(Step):
    def __init__(self, inputs, outputs, name=None):
        super(Step, self).__init__(name=name)

        inputs = listify(inputs)
        if not is_data_list(inputs):
            raise ValueError('inputs must be of type Data.')
        if len(set(inputs)) != len(inputs):
            raise ValueError('inputs must be unique.')

        outputs = listify(outputs)
        if not is_data_list(outputs):
            raise ValueError('outputs must be of type Data.')
        if len(set(outputs)) != len(outputs):
            raise ValueError('inputs must be unique.')

        self.inputs = inputs
        self.outputs = outputs
        self._graph = self._build_graph()
        self._data = self._collect_data(self._graph)
        self._default_steps = self._get_required_steps(self._graph, self.inputs, self.outputs)
        # TODO: Implement a steps_cache keyed by tuple(tuple(inputs), tuple(outputs)).
        # inputs and outputs must be sortable (implement __lt__, etc)

    def _build_graph(self):
        # Model uses the DiGraph data structure to store and operate on its Data and Steps.
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
    def _collect_data(graph):
        data = set()
        for step in graph:
            for output in step.outputs:
                data.add(output)
        return data

    @staticmethod
    def _get_required_steps(graph, inputs, outputs):
        all_steps_sorted = graph.topological_sort()  # Fail early if graph is acyclic

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
            if graph.in_degree(step) == 0:
                missing_inputs.extend(step.outputs)

        if missing_inputs:
            raise RuntimeError('The following inputs are required but were not specified:\n'
                               '{}'.format(','.join([input.name for input in missing_inputs])))

        # Check for any unused inputs
        for input in inputs:
            if input not in inputs_found:
                raise RuntimeError(
                    'Input {} was provided but it is not required to compute the specified outputs.'.format(input.name))

        return [step for step in all_steps_sorted if step in all_required_steps]

    def _normalize_given_data(self, given_data, kind) -> Dict[Data, ArrayLike]:
        if isinstance(given_data, dict):
            input_data_norm = self._normalize_dict(given_data)
        else:
            dataobjs = self.inputs if kind == 'input' else self.outputs
            expand_none = kind == 'target'
            input_data_norm = self._normalize_list(given_data, dataobjs, expand_none)

        return input_data_norm

    def _normalize_required_outputs(self, outputs: Union[str, Data, List[str], List[Data]]) -> List[Data]:
        if outputs is None:
            return self.outputs

        outputs = listify(outputs)
        outputs_norm = []
        for output in outputs:
            output = self._check_data(output)
            outputs_norm.append(output)
        return outputs_norm

    def _normalize_dict(self, data_dict: Union[Dict[Data, ArrayLike], Dict[str, ArrayLike]]) -> Dict[Data, ArrayLike]:
        data_dict_norm = {}
        for key, value in data_dict.items():
            key = self._check_data(key)
            data_dict_norm[key] = value
        return data_dict_norm

    def _normalize_list(self, data_list: Union[ArrayLike, List[ArrayLike]], dataobjs, expand_none) -> Dict[Data, ArrayLike]:
        # User passed either an array-like directly (case of one input)
        # or passed a list of array-like.
        # In this case, it must match the number of inputs specified at instantiation
        if data_list is None and expand_none:
            data_list = [None] * len(dataobjs)

        data_list = listify(data_list)

        # if len(data_list) != len(dataobjs):
        #     raise ValueError('The number of training data arrays does not match the number of inputs!\n'
        #                      'Expected {} but got {}'.format(len(dataobjs), len(data_list)))

        input_data_norm = dict(zip(dataobjs, data_list))

        return input_data_norm

    def _check_data(self, data: Union[str, Data]) -> Data:
        # Steps are assumed to have unique names
        # If the step names are unique, so are the data names
        if isinstance(data, str):
            for d in self._data:
                if data == d.name:
                    return d

        elif isinstance(data, Data):
            if data in self._data:
                return data

        raise ValueError('{} was not found in the model!'.format(data))

    def fit(self, input_data, target_data=None):
        # TODO: add extra_targets keyword argument
        # TODO: Add **fit_params argument (like sklearn's Pipeline.fit)
        # TODO: Consider using joblib's Parallel and Memory classes to parallelize and cache computations
        # In graph parlance, the 'parallelizable' paths of a graph are called 'disjoint paths'
        # https://stackoverflow.com/questions/37633941/get-list-of-parallel-paths-in-a-directed-graph
        input_data = self._normalize_given_data(input_data, 'input')

        for input in self.inputs:
            if input not in input_data:
                raise ValueError('Missing input {}'.format(input))

        target_data = self._normalize_given_data(target_data, 'target')

        for output in self.outputs:
            if output not in target_data:
                raise ValueError('Missing output {}'.format(output))

        steps = self._get_required_steps(self._graph, self.inputs, self.outputs)

        cache = dict()  # keys: Data instances, values: actual data (e.g. numpy arrays)
        cache.update(input_data)
        # cache.update(extra_targets)

        for step in steps:
            # 1) Fit phase
            Xs = [cache[i] for i in step.inputs]
            if hasattr(step, 'fit'):
                # Filtering out None target_data allow us to define fit methods without y=None.
                ys = [target_data[o] for o in step.outputs if o in target_data and target_data[o] is not None]
                step.fit(*Xs, *ys)

            # 2) predict/transform phase
            self._compute_step(step, Xs, cache)

    def predict(self, input_data, outputs=None):
        cache = dict()  # keys: Data instances, values: actual data (e.g. numpy arrays)

        input_data = self._normalize_given_data(input_data, 'input')
        outputs = self._normalize_required_outputs(outputs)
        steps = self._get_required_steps(self._graph, input_data.keys(), outputs)

        cache.update(input_data)

        for step in steps:
            Xs = [cache[i] for i in step.inputs]
            self._compute_step(step, Xs, cache)

        output_data = [cache[o] for o in outputs]
        if len(output_data) == 1:
            return output_data[0]
        else:
            return output_data

    @staticmethod
    def _compute_step(step, Xs, cache):
        # TODO: Check that number and shape of inputs/outputs is equal to the expected number and shapes
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

    # TODO: Implement build_output_shapes method.
    # TODO: Override __call__ method
    # query: inputs (outputs) can be a dictionary keyed by Data instances or
    # their names, with array values. We need input normalization for this.
    # Also, check that all of requested output keys exist in the Model graph.

    @property
    def graph(self):
        return self._graph
