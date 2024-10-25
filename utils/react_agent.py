from vertexai.generative_models import GenerationConfig, GenerativeModel, Content, GenerativeModel, Part  
from google.cloud import aiplatform  
from IPython.display import Markdown,display,HTML
import sys
import json
import os
import importlib
import math
import html

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

import rag
importlib.reload(rag)
from rag import RAG

REACT_SYSTEM_PROMPT = """
you are a financial expert and an underwriter working for the commerical banking sector of the bank.
You operate by running a loop with the following steps: Thought, Action, Observation.
You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don' make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>, "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools / actions:

<tools>
%s
</tools>

Example session number 1:
<question>What's the current temperature in Madrid?</question>
<thought>I need to get the current weather in Madrid</thought>
<tool_call>{"name": "get_current_weather","arguments": {"location": "Madrid", "unit": "celsius"}, "id": 0}</tool_call>
You will be called again with this:
<observation>{0: {"Retrieved data": f"the temperature is 25 Celsius"}}</observation>
Now you have the answer, you then output:
<answer>The current temperature in Madrid is 25 degrees Celsius</answer>

Example session number 2:
<question>analyze financial performance of Apple Inc in 2023 and potential drivers</question>
<thought>to analyse finanial performance, I need to know Apple's total revenue, so I need to search its total revenue in my knowledge base</thought>
<tool_call>{"name": "search","arguments": {"query": "Apple Inc total revenue 2023"}, "id": 0}</tool_call>
You will be called again with this:
<observation>{0: {"Retrieved data": "Total Revenue 123,456......"}}</observation>
You then run another iteration:
<thought>to know the potential drivers, I need to know the buisness compositions of Apple Inc, so I need to search its buisness compositions</answer>
<tool_call>{"name": "search","arguments": {"query": "Apple Inc buisness composition"}, "id": 0}</tool_call>
You will be called again with this:
<observation>{0: {"Retrieved data": "Apple's business composition includes iPhone: Major revenue driver. Mac: Laptops and desktops......"}}</observation>
Now you have the answer, you then output:
<answer>Apple's financial performance is strong with a total revenue of 123,456, which is drived by iPhone sales.</response>


Additional constraints:

- If the user asks you something unrelated to any of the tools above, answer the question freely using your own knowledge, enclosing your answer with <answer></answer> tags.
"""

def print_in_color(text, color, escape=False):
    if escape:
        text = html.escape(text)
    display(HTML(f'<span style="color: {color};">{text}</span>'))
    
class ReactAgent:
    """
    A class that represents an agent using the ReAct logic that interacts with tools to process
    user inputs, make decisions, and execute tool calls. The agent can run interactive sessions,
    collect tool signatures, and process multiple tool calls in a given round of interaction.
    """

    def __init__(self, tools: Tool | list[Tool], system_prompt : str = None, print_system_prompt = False, model='gemini-1.5-pro-002') -> None:  
        
        self.client = GenerativeModel(model)
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.system_prompt =  REACT_SYSTEM_PROMPT % self.add_tool_signatures()
        if system_prompt:
            self.system_prompt=system_prompt
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
            print_in_color(text=f"\n Tool calling details: \n{validated_tool_call}", color='green')

            # run the tool
            result = actual_tool.run(**validated_tool_call["arguments"])
            
            
            # add the result to the observations map using id, because there might be multiple tools to execute
            observations[validated_tool_call["id"]] = result

        return observations
            
    def run(self, user_msg: str, max_iterations: int = 10,) -> str:
                        
        chat_history = ChatHistory(
            [
                create_single_text_Content(
                    text=self.system_prompt,
                    role="user",
                    
                ),
                create_single_text_Content(
                    text=user_msg, 
                    role="user",
                    added_tag='question'
                ),
            ]
        )
        
        if self.print_system_prompt:
            print(self.system_prompt)
            
        for i in range(max_iterations):
            print('-'*40,f'iteration {i} ', '-'*40)
            response = execute(self.client, messages=chat_history)   
            
            print_in_color(text="thought and action\n:"+str(response), color='red', escape=True)     
            
            answer = extract_tag_content(str(response), "answer")
            if answer.found:
                return answer.content[0]
            
            # thought = extract_tag_content(str(response), "thought")
            tool_calls = extract_tag_content(str(response), "tool_call")
            update_chat_history(history=chat_history, msg=response, role="model")
            
            if tool_calls.found:
                observations = self.process_tool_calls(tool_calls.content)
                print_in_color(text=f"observation:\n", color='blue')
                for observation in observations.values():
                    print(observation)
                update_chat_history(history=chat_history, msg=observations, role="user", added_tag='observation')
            else:
                temp_msg="\nObservations: tool not found, think again, choose another tool"
                print_in_color(text=f"{temp_msg}", color='blue')
                update_chat_history(history=chat_history, msg=temp_msg, role="user", added_tag='observation')
                        
        print("max iterations reached")
        return chat_history
