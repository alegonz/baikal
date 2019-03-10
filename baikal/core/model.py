from inspect import signature

from baikal.core.data import is_data_list
from baikal.core.step import Step
from baikal.core.utils import listify


class Model(Step):
    def __init__(self, inputs, outputs, name=None):
        super(Step, self).__init__(name=name)

        inputs = listify(inputs)
        outputs = listify(outputs)

        if not is_data_list(inputs) or not is_data_list(outputs):
            raise ValueError('inputs and outputs must be of type Data.')

        self.inputs = inputs
        self.outputs = outputs
        self._steps = self._get_required_steps()

    def _get_required_steps(self):
        all_steps_sorted = self.graph.topological_sort()

        # Backtrack from outputs until inputs to get the necessary steps. That is,
        # find the ancestors of the nodes that provide the specified outputs.
        # Raise an error if there is an ancestor whose input is not in the specified inputs.
        # We assume a DAG (guaranteed by success of topological_sort) and with no
        # multiedges (guaranteed by success of add_edge).

        required_steps = set()

        # We need to compute the step associated with each output and its ancestors
        for output in self.outputs:
            required_steps.add(output.step)
            required_steps |= self.graph.ancestors(output.step)

        # We do not need to compute the step associated with each input and its ancestors
        for input in self.inputs:
            required_steps.remove(input.step)
            required_steps -= self.graph.ancestors(input.step)

        return [step for step in all_steps_sorted if step in required_steps]

    def fit(self, input_data, target_data=None):
        # TODO: add extra_targets keyword argument
        # TODO: Add **fit_params argument (like sklearn's Pipeline.fit)
        # TODO: Consider using joblib's Parallel and Memory classes to parallelize and cache computations
        # In graph parlance, the 'parallelizable' paths of a graph are called 'disjoint paths'
        # https://stackoverflow.com/questions/37633941/get-list-of-parallel-paths-in-a-directed-graph

        input_data = listify(input_data)
        target_data = listify(target_data)

        cache = dict()  # keys: Data instances, values: actual data (e.g. numpy arrays)
        cache.update(zip(self.inputs, input_data))  # FIXME: Raise error if lengths do not match
        cache.update(zip(self.outputs, target_data))  # FIXME: Raise error if lengths do not match

        for step in self._steps:
            # 1) Fit step
            #     check signature of fit method
            #     if fit only needs X, retrieve it from cache
            #     it fit also needs y, retrieve it from extra_targets
            Xs = [cache[i] for i in step.inputs]

            if 'y' in signature(step.fit).parameters:
                ys = [cache[o] for o in step.outputs]
                output_data = step.fit(*Xs, *ys)
            else:
                output_data = step.fit(*Xs)

            # 2) Compute outputs
            #     predict if step is an estimator
            #     transform if step is a transformer

    # TODO: Implement compute method. Should call predict inside.
    # TODO: Implement build_output_shapes method. Should call predict inside.
    # TODO: Override __call__ method
    # TODO: Implement predict method
    # predict: inputs (outputs) can be either: a list of arrays
    # (interpreted as 1to1 correspondence with inputs (outputs) passed at __init__),
    # or a dictionary keyed by Data instances or their names with array values. We need input normalization for this.
