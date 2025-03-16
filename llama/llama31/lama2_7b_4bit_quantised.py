# Just to test LLAMA2 quantised models

from lib_llm import LLAMa2Q

usercontext ='''

Given the Context below

"""

Cholesterol is a waxy substance found in your blood. Your body needs cholesterol to build healthy cells, but high levels of cholesterol can increase your risk of heart disease.

With high cholesterol, you can develop fatty deposits in your blood vessels. Eventually, these deposits grow, making it difficult for enough blood to flow through your arteries. Sometimes, those deposits can break suddenly and form a clot that causes a heart attack or stroke.

High cholesterol can be inherited, but it's often the result of unhealthy lifestyle choices, which make it preventable and treatable. A healthy diet, regular exercise and sometimes medication can help reduce high cholesterol.

Answer the following question if possible
What is the role of cholestrol in the body
'''


if __name__ == "__main__":
    print("Going to load the llama2 7b model ...")
    #llam2_4bit = LLAMa2Q(model_name="meta-llama/Llama-2-7b-chat-hf",q4bitA=True)
    llam2_4bit = LLAMa2Q(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",q4bitA=True)
    
    print("Loaded LLama2 7b model")

    prompt = f"{ usercontext}"
    system_message = "You are a helpful assistant.Please answer the question if it is possible from the context"
    prompt_template=f'''[INST] <<SYS>>
    {system_message}
    <</SYS>>

    {prompt} [/INST]'''

    prompt_template = llam2_4bit.create_prompt(prompt,system_message)
    output = llam2_4bit.generate_ouputput(prompt_template)
    response =llam2_4bit.get_chat_response(output)
    print(response)