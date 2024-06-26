import logging
import logging.config
import os
from typing import Dict, List, Union

import numpy as np
from base import SearchClass

try:
    logging.config.fileConfig(os.path.join(os.getcwd(), "ee-one-d", "logging.conf"))
except Exception as e:
    logging.error("Cwd must be root of project directory")
logger = logging.Logger(__name__)


class SearchPipeline:

    def __init__(
        self,
        input_class_list: List[SearchClass],
        primary_arg: str,
        primary_arg_name: str,
        init_arg_dict: List[Dict],
        call_arg_dict: List[Dict],
    ) -> None:

        logger.debug(
            f"Initializing SearchPipeline class. \
                      input_class_list: {input_class_list}, \
                      primary_arg: {primary_arg}, \
                      primary_arg_name: {primary_arg_name}, \
                      init_arg_dict: {init_arg_dict}, \
                      call_arg_dict: {call_arg_dict}"
        )

        self.input_class_list = input_class_list
        self.primary_arg = primary_arg
        self.primary_arg_name = primary_arg_name
        self.init_arg_dict = init_arg_dict
        self.call_arg_dict = call_arg_dict

    def run(self) -> List:
        logger.debug("Running SearchPipeline")
        results = []
        while self.input_class_list and self.init_arg_dict and self.call_arg_dict:

            current_class, current_init_args, current_call_args = (
                self.input_class_list.pop(0),
                self.init_arg_dict.pop(0),
                self.call_arg_dict.pop(0),
            )
            if not results:

                logger.debug(f"Initializing  {current_class.__name__}")
                current_init_args[self.primary_arg_name] = self.primary_arg

                for result in np.reshape(
                    np.array([current_class(**current_init_args)(**current_call_args)]),
                    (-1),
                ):
                    results.append(result)

                logger.debug(f"Results: {results}")

            else:

                new_results = []
                while results:

                    logger.debug(f"Calling {current_class.__name__}")
                    current_result = results.pop(0)
                    logger.debug(f"Current result {current_result}")

                    current_init_args[self.primary_arg_name] = current_result

                    for result in np.reshape(
                        np.array(
                            [current_class(**current_init_args)(**current_call_args)]
                        ),
                        (-1),
                    ):
                        new_results.append(result)

                    logger.info(f"New results: {new_results}")

                results = new_results

        return results


if __name__ == "__main__":

    class ClassA:

        def __init__(self, input_value):
            self.input_value = input_value

        def __call__(self, add_value):
            return self.input_value + add_value

    class ClassB:

        def __init__(self, input_value):
            self.input_value = input_value

        def __call__(self, multiply_value):
            return self.input_value * multiply_value

    class ClassC:

        def __init__(self, input_value):
            self.input_value = input_value

        def __call__(self, subtract_value):
            return self.input_value - subtract_value

    # Example initialization and call arguments
    input_class_list = [ClassA, ClassB, ClassC]
    call_arg_dict = [{"add_value": 10}, {"multiply_value": 2}, {"subtract_value": 3}]
    init_arg_dict = [
        {},
        {},
        {},
    ]

    # Create TextPipeline instance
    pipeline = SearchPipeline(
        input_class_list,
        primary_arg=3,
        primary_arg_name="input_value",
        init_arg_dict=init_arg_dict,
        call_arg_dict=call_arg_dict,
    )

    results = pipeline.run()
    print("Final results:", results)
