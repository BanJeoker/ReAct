from vertexai.generative_models import Content, Part  

def execute(client, messages: list, generation_config={"temperature":0}) -> str:
    """
    Sends a request to the client's `generate_content` method to interact with the language model.

    Args:
        client (Gemini model): The gemini model
        messages (list[Content]): A list of Content as chat history
        generation_config: generation configuration
        
    Returns:
        str: The content of the model's response.
    """
    response = client.generate_content(contents=messages, generation_config=generation_config) 
    return response.text

def add_tag_to_text(text, tag):

    results = f"<{tag}>{text}</{tag}>"
    return results

def create_single_text_Content(role, text, added_tag=None):
    '''
    Create a Content object for history, containing only one part, which is a text
    Args:
        role: can be either 'user' or 'model'
        text: the content, the type can be integer, float, or str, since executed results from our functions can return many types.
    Return: Content
    '''
    
    if added_tag:
        text=add_tag_to_text(text,added_tag)
    else:
        # make sure the final text to be appended is str, because we are using Part.from_text()
        text=str(text)
        
    return Content(role=role, parts=[Part.from_text(text)])
    

def update_chat_history(history: list, msg, role: str, added_tag=None, msg_type='regular'):
    """
    Updates the chat history by appending the latest response. could be (1) the prompt, (2) the initial query, or (3) observations

    Args:
        history (list): The list representing the current chat history.
        msg: the msg to be appended
        role (str): The role type ('user' or 'model')
    """
    
    """
    Normally, we return a string back to llm as the message. We can combine int, double, json string all to a single string as the message to send to LLM. However, we can also send a Part to LLM, which is not a string, but a Gemini specific object. In that case, we need to do more manual customizaiton as shown below. Codes here can definitely be improved.
    
    the strucutre of the history for gemini can be found here https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_chat.ipynb (In [11])
    
    history=[
        Content(user='model', parts=[Part, Part, Part,...]),
        Content(user='user', parts=[Part]),
        Content(user='model', parts=[Part, Part])
    ]
    
    history is made of a list of Content, each Content has the 'user', and 'parts', 'parts' is a list of Part.
    to mix and match:
        we can use Part.from_text(text) to convert a string to a Part
        we can use Part.from_data() to convert a input pdf to a Part
    
    """
    
    
    # if the msg is regular printable objects, we can just convert the entire msg dict to a string
    if msg_type=='observation_regular_printable' or msg_type=='regular':
        history.append(create_single_text_Content(text=msg, role=role, added_tag=added_tag))
        
    # if the full pdf as a Part is returned, we have to customize the Content as a list of Parts
    elif msg_type=='observation_part':
        heading_part=Part.from_text('<observation></observation> follows:')
        parts=[heading_part]
         # the msg must be a dict containing the results returned from function calls. the key of the dict is function call ids, since there can be multiple function calls.
        for part in msg.values():
            parts.append(part)
        history.append(Content(role=role, parts=parts))
        

class ChatHistory(list):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """
        Initialise the list with a fixed total length.

        Args:
            messages (list | None): A list of initial messages, each message is a Content
            total_length (int): The maximum number of messages the chat history can hold.
        """
        if messages is None:
            messages = []

        super().__init__(messages)
        self.total_length = total_length

    def append(self, msg):
        """Add a Content to the queue.
        Args:
            msg (Content): The message to be added to the queue
        """
        if len(self) == self.total_length:
            self.pop(0)
        super().append(msg)
        