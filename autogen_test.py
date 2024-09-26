"""
Run Server like
python3 -m llama_cpp.server --model /home/alex/coding/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf   --host localhost --port 8080
Run ollama directly
ssd/llama.cpp$ ./main -m ~/coding/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf --color --ctx_size 2048 -n -1 -ins -b 256 --top_k 10000 --temp 0.2 --repeat_penalty 1.1 -t 8 -ngl 10000  --multiline-input
"""


import autogen
import pylint.lint
from pylint.reporters.text import TextReporter
from io import StringIO
import os

 #------------------------------------------------------------------------Configurations-------------------------------------

config_list_low = [
    {
        "model":"mistral-instruct-v0.2",
        "base_url": "http://localhost:8080/v1",
        "api_key":"NULL"
    }
    ,{
        "model": "gpt-3.5-turbo-1106",  # model name
        "api_key": ""  # api key
    }
]

high_config = {
        "cache_seed": 442,
        "temperature": 0,
        "config_list":  [
        {
                "model": "gpt-4-0613",  # model name
                "api_key": ""  # api key
         }
        ],
    }


#----------------------------------------Agents and Agent Proxies ------------------------------------------------------------------------------------

developer_agent = autogen.AssistantAgent(
    name="developer_agent",
    system_message="""You are a helpful code reivew assistant.\
    Use code_linter first to get the pylint errors to help in review.
    Use ask_expert function with your review comments for a more thorough review.Reply TERMINATE when the task is done.""",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list_low,
        "temperature": 0,
        "functions": [
            {
                "name": "ask_expert",
                "description": "An Expert Code Review agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "your review comments",
                        },
                        "code_snippet": {
                            "type": "string",
                            "description": "original code snippet",
                        },
                    },
                    "required": ["code_snippet","message"],
                },
            },
            {
                "name": "code_linter",
                "description": "Use Pylint as a tool for intial review",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code_snippet": {
                            "type": "string",
                            "description": "The code snippet to review",
                        },
                    },
                    "required": ["code_snippet"],
                },
            }
        ],
    },
)


#----------------------------------------Functions ------------------------------------------------------------------------------------

def lint_code_snippet(code_snippet):
    """Lints a code snippet using pylint."""

    temp_file_path = 'temp.py'
    # Create a temporary file to write the code snippet to, as PyLinter requires file paths
    with open(temp_file_path, 'w') as temp_file:
        temp_file.write(code_snippet)
     
    # Setup the in-memory output stream for pylint reports
    output = StringIO()
    
    # Initialize the linter
    linter = pylint.lint.PyLinter(reporter=TextReporter(output=output))
    linter.load_default_plugins()  # Load the default pylint plugins
    linter.check([temp_file_path])
    
    os.remove(temp_file_path)
    # Return the captured output
    return output.getvalue()


def ask_expert(code_snippet,message):

    print(f"In ask_expert code_snippet={code_snippet} message={message}")

    expert_agent = autogen.AssistantAgent(
        name="expert_agent",
        system_message="""you are a helpful assistant highly skilled in evaluating given code by providing a score from 1 (bad) - 10 (good) 
        while providing clear rationale.
        Specifically, you can carefully evaluate the code across the following dimensions
        - bugs (bugs):  are there bugs, logic errors, syntax error or typos? 
        - performance (optimization): is the code effecient? can you suggest tips for better performance ?
        - security (compliance): Are good security practices followed. Give score as NA if this is not applicable
        - readability/maintainability:: How readable and maintainable is the code
        YOU MUST PROVIDE A SCORE for each of the above dimensions froma score from 1 (bad) - 10 (good)
        {bugs: 0, performance: 0, security: 0, readability/maintainability: 0}
        Finally, based on the critique above, suggest a concrete list of actions that the coder should take to improve the code.
        """,
        llm_config=high_config
    )
    expert = autogen.UserProxyAgent(
        name="expert",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        code_execution_config= False, # disable code executiong
    )
    
    expert.initiate_chat(expert_agent,message=f"For the followin code \n '{code_snippet}' \n these are the initial code review comments " +
     f"along with pylint findings  \n '{message}'")
    #expert.stop_reply_at_receive(expert_agent)
    # final message sent from the expert
    expert.send(f"Summarise and show code snippets for the comments if possible", expert_agent)
    # return the last message the expert received
    return expert.last_message()["content"]

developer = autogen.UserProxyAgent(
    name="developer",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    code_execution_config={
        "work_dir": "developer",
        "use_docker": False,
    }, 
    function_map={"code_linter": lint_code_snippet,"ask_expert": ask_expert}
)


#-----------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__": 

    print ("Autogen Bots for Code Review")

    code_snippet="""
    for message in messages:
        email_data = dict()
        msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()

        headers = msg['payload']['headers']
        date = next(header['value'] for header in headers if header['name'] == 'Date')
        print(f"Date: {date}")
        email_data["date"]= date
    """

    # the assistant receives a message  with code review parts
    developer.initiate_chat(
        developer_agent,
        message=code_snippet,
    )