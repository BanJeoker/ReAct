from vertexai.generative_models import GenerationConfig, GenerativeModel, Content, GenerativeModel, Part  
from google.cloud import aiplatform  
from IPython.display import Markdown,display,HTML
import sys
import json
import os
import importlib
import math
import html
import time

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
Now, based on the observation, you have the answer, you then output:
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
Now, based on the observation, you have the answer, you then output:
<answer>Apple's financial performance is strong with a total revenue of 123,456, which is drived by iPhone sales.</response>


Additional constraints:
- You do not output <observation></observation> tags, observations are provided to you.
- When you have the answer, always enclosing your answer with <answer></answer> tags.
- You are running in a infinite loop, so do not give up so easily by saying you cannot answer the quetsion. Keep using the tools provided for you to find solutions.
"""

def color_box(text, color, title):
    
    text_color='black'
    color_dict={
        'red': ["rgba(255, 0, 0, 0.1)", "rgba(255, 0, 0, 0.9)","red"],
        'blue': ["rgba(56, 113, 224, 0.1)", "rgba(56, 113, 224, 0.9)","blue"],
        'green': ["rgba(0, 255, 0, 0.1)", "green","green"]
    }
    
    background_color=color_dict[color][0]
    border_color=color_dict[color][1]
    title_color=color_dict[color][2]
    html_box = f"""
    <div style="
        padding: 5px;
        border-radius: 5px;
        color: {text_color};
        background-color: {background_color};
        border: 2px solid {border_color};
        font-size: 13px;
    ">
    <strong style="color: {title_color}; font-size: 20px;">{title}</strong><br>
    <pre style="font-family: inherit; font-size: inherit; color: {text_color}; background: transparent; margin: 0;">{html.escape(text)}</pre>
    </div>
    """
    return HTML(html_box)
    
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
        self.chat_history=None
            
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
            
            
            # get the actual tool, the physical existing function
            actual_tool = self.tools_dict[tool_name]

            # validate or transform the argument, for example, if the function need integer, we have to manually convert numbers in string returned by LLM to the actual integer        
            validated_tool_call = validate_arguments(
                tool_call, json.loads(actual_tool.fn_signature)
            )
            
            display(color_box(f"Tool calling details: \n{validated_tool_call}", color='green',title=f'Parsed Tool is: {tool_name}'))

            # run the tool
            result = actual_tool.run(**validated_tool_call["arguments"])
            
            
            # add the result to the observations map using id, because there might be multiple tools to execute; the result can be of different types of objects, most commonly string. But in the case of returning the who pdf, a Part object read from from_data() method can also be expected.
            observations[validated_tool_call["id"]] = result

        return observations
            
    def run(self, query: str, max_iterations: int = 10,) -> str:
                        
        self.chat_history = ChatHistory(
            [
                create_single_text_Content(
                    text=self.system_prompt,
                    role="user",
                    
                ),
                create_single_text_Content(
                    text=query, 
                    role="user",
                    added_tag='question'
                ),
            ]
        )
        
        if self.print_system_prompt:
            print(self.system_prompt)
            
        for i in range(max_iterations):
            print()
            print('-'*40,f'iteration {i} ', '-'*40)
            response = execute(self.client, messages=self.chat_history)   
            
            # print thought and action
            display(color_box(text=str(response), color='red',title='Thought and Action'))
            
            # parse the answer, if answer is present
            answer = extract_tag_content(str(response), "answer")
            if answer.found:
                return answer.content[0]
            
            # parse the actions
            tool_calls = extract_tag_content(str(response), "tool_call")
            update_chat_history(history=self.chat_history, msg=response, role="model")
            
            # execute the actions and returns the observation
            if tool_calls.found:
                
                # execute the function, returned results are observations
                observations = self.process_tool_calls(tool_calls.content)
                
                # print the observations
                msg_type='observation_regular_printable'
                for observation in observations.values():
                    # the executed results can be of many types, depends on the specific function used
                    if isinstance(observation, Part): # edge case
                        msg_type='observation_part'
                        # print('observation is non-printable Part object, probably the full pdf')
                        display(color_box(text='observation is non-printable Part object, probably the full pdf', color='blue',title='Observation'))
                    else: 
                        display(color_box(text=observation, color='blue',title='Observation'))
                        
                # update the chat history with the newest observations
                update_chat_history(history=self.chat_history, msg=observations, role="user", added_tag='observation', msg_type=msg_type)
                
            else: #if tool not found
                temp_msg="\nObservations: tool not found, think again, choose another tool"
                display(color_box(text=temp_msg, color='blue',title='Observation'))
                update_chat_history(history=self.chat_history, msg=temp_msg, role="user", added_tag='observation')
            time.sleep(2)
                        
        print("max iterations reached", '!'*10)
