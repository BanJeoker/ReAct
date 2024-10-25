from vertexai.generative_models import GenerationConfig, GenerativeModel, Content, GenerativeModel, Part  
from google.cloud import aiplatform  
from IPython.display import Markdown,display,HTML
import sys
import json
import os
import importlib
sys.path.append(os.path.abspath('../utils'))

import chat_functions
importlib.reload(chat_functions)
from chat_functions import execute, create_single_text_Content,update_chat_history, ChatHistory  

import parser
importlib.reload(parser)
from parser import extract_tag_content

import tool_functions
importlib.reload(tool_functions)
from tool_functions import get_fn_signature, validate_arguments, Tool, convert_to_tool


SYSTEM_PROMPT= """
You are a function calling and questions answering AI model. You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

For each function call return a json object with function name and arguments within <tool_call></tool_call>
XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>,  "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools:

<tools>
%s
</tools>

if none of the functions is related to the question, answer the question using your own knowledge.
if you are provided with observations, you can answer the question based on that observation, that observation is often the executed results from the tools you just selected.
"""

def print_in_color(text, color):
    display(HTML(f'<span style="color: {color};">{text}</span>'))

class ToolAgent:
    
    def __init__(self, tools: Tool | list[Tool], print_system_prompt=False) -> None:
        
        self.client = GenerativeModel("gemini-1.5-pro-002")
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.print_system_prompt=print_system_prompt
            
    def add_tool_signatures(self) -> str:
        """
        Collects the function signatures of all available tools.

        Returns:
            str: A concatenated string of all tool function signatures in JSON format.
        """
        return "".join([tool.fn_signature for tool in self.tools])

    def process_tool_calls(self, tool_calls_content: list) -> dict:
        """
        Processes each tool call, validates arguments, executes the tools, and collects results.

        Args:
            tool_calls_content (list): List of strings, each representing a tool call in JSON format.

        Returns:
            dict: A dictionary where the keys are tool call IDs and values are the results from the tools.
        """
        observations = {}
        # again, there might be several tool_calls
        
        for tool_call_str in tool_calls_content:
            
            tool_call = json.loads(tool_call_str)

            # get the parased tool name
            tool_name = tool_call["name"]            
            print_in_color(text=f"\n Parsed Tool is: {tool_name}", color='green')
            
            # get the actual tool, the physical existing function
            actual_tool = self.tools_dict[tool_name]

            # validate or transform the argument, for example, if the function need integer, we have to manually convert numbers in string returned by LLM to the actual integer        
            validated_tool_call = validate_arguments(
                tool_call, json.loads(actual_tool.fn_signature)
            )
            print_in_color(text=f"\nTool call dict: \n{validated_tool_call}", color='green')

            # run the tool
            result = actual_tool.run(**validated_tool_call["arguments"])

            print_in_color(text=f"\n Result to be added as observation: \n{result}", color='green')
            
            # add the result to the observations map using id, because there might be multiple tools to execute
            observations[validated_tool_call["id"]] = result

        return observations
    
    def run(self, user_msg: str,):
        """
        Handles the full process of interacting with the language model and executing a tool based on user input.

        Args:
            user_msg (str): The user's message that prompts the tool agent to act.

        Returns:
            str: The final output after executing the tool and generating a response from the model.
        """
            
        chat_history = ChatHistory(
            [
                create_single_text_Content(
                    text=SYSTEM_PROMPT % self.add_tool_signatures(),
                    role="user",
                ),
                create_single_text_Content(
                    text=user_msg, 
                    role="user"
                ),
            ]
        )
        
        if self.print_system_prompt:
            print(chat_history)
        
        response = execute(self.client, messages=chat_history)
        tool_calls = extract_tag_content(str(response), "tool_call")
        
                       
        if tool_calls.found:
            # tool_calls.content is a list, since there might be several tool calls 
            observations = self.process_tool_calls(tool_calls.content)
            update_chat_history(history=chat_history, msg=f"Observation: {observations}", role="user")
        
        final_answer=execute(self.client, messages=chat_history)
                
        return final_answer