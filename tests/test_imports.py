def test_model_and_step_imports():
    from baikal import Model
    from baikal import Input
    from baikal import Step

    from baikal.steps import Step
    from baikal.steps import Input

    from baikal.steps import Lambda, Concatenate, Stack
    from baikal.steps.expression import Lambda
    from baikal.steps.merge import Concatenate, Stack
