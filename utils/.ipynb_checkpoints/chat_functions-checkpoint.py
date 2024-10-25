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
        text: the content
    Return: Content
    '''
    if added_tag:
        text=add_tag_to_text(text,added_tag)
    
    return Content(role=role, parts=[Part.from_text(text)])
    

def update_chat_history(history: list, msg: str, role: str, added_tag=None):
    """
    Updates the chat history by appending the latest response.

    Args:
        history (list): The list representing the current chat history.
        msg (str): The message to append.
        role (str): The role type ('user' or 'model')
    """
    history.append(create_single_text_Content(text=msg, role=role, added_tag=added_tag))


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
        