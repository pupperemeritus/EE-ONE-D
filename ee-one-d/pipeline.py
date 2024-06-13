from typing import Dict, List, Union

import numpy as np
from base import SearchClass


class SearchPipeline:
    def __init__(
        self,
        input_class_list: List[SearchClass],
        primary_arg: str,
        primary_arg_name: str,
        init_arg_dict: List[Dict],
        call_arg_dict: List[Dict],
    ):
        """
        Initializes the SearchPipeline with input parameters.

        Parameters
        ----------
        input_class_list : list of SearchClass
            A list of SearchClass instances.
        primary_arg : str
            The primary argument.
        init_arg_dict : list of dict
            A list of dictionaries for initialization arguments.
        call_arg_dict : list of dict
            A list of dictionaries for call arguments.
        """
        self.classes = input_class_list
        self.primary_arg = primary_arg
        self.primary_arg_name = primary_arg_name
        self.init_arg_list = init_arg_dict
        self.call_arg_list = call_arg_dict
        self.objects = []

    def _initialize_classes(self):
        """
        Initialize all classes in input_class_list with respective init_arg_list.
        """
        for class_type, init_args in zip(self.classes, self.init_arg_list):
            self.objects.append(class_type(**init_args))

    def run(self):
        """
        Execute the pipeline by initializing classes and sequentially calling their __call__ methods.
        """
        self._initialize_classes()
        results = []
        curr_input = self.primary_arg

        for class_obj, call_args in zip(self.objects, self.call_arg_list):
            # Pack curr_input into call_args before calling class_obj
            call_args_with_input = {**call_args, self.primary_arg_name: curr_input}
            curr_input = class_obj(**call_args_with_input)
            results.append(curr_input)

        return results


if __name__ == "__main__":

    class ClassA:
        def __init__(self, add_value):
            self.add_value = add_value

        def __call__(self, input_value):
            return input_value + self.add_value

    class ClassB:
        def __init__(self, multiply_value):
            self.multiply_value = multiply_value

        def __call__(self, input_value):
            return input_value * self.multiply_value

    class ClassC:
        def __init__(self, subtract_value):
            self.subtract_value = subtract_value

        def __call__(self, input_value):
            return input_value - self.subtract_value

    # Example initialization and call arguments
    input_class_list = [ClassA, ClassB, ClassC]
    init_arg_dict = [{"add_value": 10}, {"multiply_value": 2}, {"subtract_value": 3}]
    call_arg_dict = [
        {},
        {},
        {},
    ]

    # Create TextPipeline instance
    pipeline = SearchPipeline(
        input_class_list,
        primary_arg=2,
        primary_arg_name="input_value",
        init_arg_dict=init_arg_dict,
        call_arg_dict=call_arg_dict,
    )

    results = pipeline.run()
    print("Final results:", results)
