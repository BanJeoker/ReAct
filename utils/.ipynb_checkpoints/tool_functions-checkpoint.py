import json
from typing import Callable



def get_fn_signature(fn: Callable) -> dict:
    """
    Generates the signature for a given function.

    Args:
        fn (Callable): The function whose signature needs to be extracted.

    Returns:
        dict: A dictionary containing the function's name, description,
              and parameter types.
    """
    fn_signature: dict = {
        "name": fn.__name__,   # name of the function 
        "description": fn.__doc__, # decription of the function
        "parameters": {"properties": {}}, # types of parameters
    }
    schema = {
        k: {"type": v.__name__} for k, v in fn.__annotations__.items() if k != "return"
    }
    fn_signature["parameters"]["properties"] = schema
    
    '''
    def say_hello(x: int, y: str) -> bool:
    say_hello.__annotations__ :{'x': int, 'y': str, 'return': bool}
    schema =   {'x': {'type': 'int'}, 'y': {'type': 'str'}}
    '''

    
    return fn_signature


def validate_arguments(tool_call: dict, tool_signature: dict) -> dict:
    """
    When LLM returns its chosen tool and arguments, we have to validate and converts arguments in the input dictionary to match the expected types. like string '2' to integer 2, if the function needs integer.

    Args:
        tool_call (dict): A dictionary containing the arguments passed to the tool.
        tool_signature (dict): The expected function signature and parameter types.

    Returns:
        dict: The tool call dictionary with the arguments converted to the correct types if necessary.
    """
    
    
#     properties = tool_signature["parameters"]["properties"]
#     # TODO: This is overly simplified but enough for simple Tools.
#     type_mapping = {
#         "int": int,
#         "str": str,
#         "bool": bool,
#         "float": float,
#     }

#     for arg_name, arg_value in tool_call["arguments"].items():
#         expected_type = properties[arg_name].get("type")

#         if not isinstance(arg_value, type_mapping[expected_type]):
#             tool_call["arguments"][arg_name] = type_mapping[expected_type](arg_value)

    return tool_call



class Tool:
    """
    A class representing a tool that wraps a callable and its signature.

    Attributes:
        name (str): The name of the tool (function).
        fn (Callable): The function that the tool represents.
        fn_signature (str): JSON string representation of the function's signature.
    """

    def __init__(self, name: str, fn: Callable, fn_signature: str):
        self.name = name
        self.fn = fn
        self.fn_signature = fn_signature

    def __str__(self):
        return self.fn_signature

    def run(self, **kwargs):
        """
        Executes the tool (function) with provided arguments.

        Args:
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The result of the function call.
        """
        return self.fn(**kwargs)



# this is the decorator    
def convert_to_tool(fn: Callable)-> Tool:
    """
    A decorator that wraps a function into a Tool object.

    Args:
        fn (Callable): The function to be wrapped.

    Returns:
        Tool: A Tool object containing the function, its name, and its signature.
    """

    fn_signature = get_fn_signature(fn)
    return Tool(
        name=fn_signature.get("name"), fn=fn, fn_signature=json.dumps(fn_signature)
    )