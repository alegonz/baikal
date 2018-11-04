def get_camelcase_humps(string):
    return ''.join(char for char in string if char.isupper() or char.isdigit())


class ArrayNode:
    _counter = 0
    _arrays = []

    def __init__(self, *, name=None):

        if name is None:
            name = ArrayNode.issue_name()

        if name in ArrayNode._arrays:
            raise ValueError('An array called {} already exists'.format(name))

        self.name = name
        ArrayNode._arrays.append(name)

    @classmethod
    def clear_arrays(cls):
        cls._arrays.clear()
        cls._counter = 0

    @classmethod
    def issue_name(cls):
        name = 'arr_{}'.format(cls._counter)
        cls._counter += 1
        return name


class ProcessorNodeMixin:
    _counter = 0
    _processors = []

    def __init__(self, *args, name=None, **kwargs):

        if name is None:
            name = self.issue_name()

        if name in ProcessorNodeMixin._processors:
            raise ValueError('A processor called {} already exists'.format(name))

        self.name = name
        ProcessorNodeMixin._processors.append(name)

        super(ProcessorNodeMixin, self).__init__(*args, **kwargs)

    @classmethod
    def clear_processors(cls):
        cls._processors.clear()
        cls._counter = 0

    def issue_name(self):
        basename = get_camelcase_humps(self.__class__.__name__)
        name = '{}_{}'.format(basename, ProcessorNodeMixin._counter)
        ProcessorNodeMixin._counter += 1
        return name
