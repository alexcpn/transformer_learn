# -*- coding: utf-8 -*-
"""mistral-7b-4bit-qa_generator_2.ipynb


Original file is located at
    https://colab.research.google.com/drive/1VE2QLYbTBWmhg_sC0jtXVpSNGA9BHC2x#scrollTo=UIPmEXF1HF7-
   
"""
import traceback
import re
import locale
locale.getpreferredencoding = lambda: "UTF-8"
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from lib_llm import MistralQ # from parent


pattern = r"(?:Q:|Question:)\s*(.*?)\n(?:A:|Answer:)\s*(.*?)(?=\n(?:Q:|Question:)|\n$)"
  
def parse_response_out (response):
  return  re.findall(pattern, response, re.DOTALL)

if __name__ == "__main__":

    system_message = ""

    filename = './output.mdx'
    #filename = '../data/small_3.txt'

    print("Going to load the Mistal 7b model ...")
    #mistral7b_4bit = MistralQ(model_name="mistralai/Mistral-7B-Instruct-v0.1",q4bitA=False)#load in 8 bit
    mistral7b_4bit = MistralQ(model_name="mistralai/Mistral-7B-Instruct-v0.1",q4bitA=True)#load in 4 bit
    print("Loaded Mistral 7b model")

    #chunks = sliding_window(filename)
chunk = """prompt=Ceph Commands
    Run any ceph command with  `kubectl rook-ceph ceph <args>`. For example, get the Ceph status:
    ```
    kubectl rook-ceph ceph status

    ```
    Output:
    ```
    cluster:
    id:     a1ac6554-4cc8-4c3b-a8a3-f17f5ec6f529
    health: HEALTH_OK

    services:
    mon: 3 daemons, quorum a,b,c (age 11m)
    mgr: a(active, since 10m)
    mds: 1/1 daemons up, 1 hot standby
    osd: 3 osds: 3 up (since 10m), 3 in (since 8d)

    data:
    volumes: 1/1 healthy
    pools:   6 pools, 137 pgs
    objects: 34 objects, 4.1 KiB
    usage:   58 MiB used, 59 GiB / 59 GiB avail
    pgs:     137 active+clean

    io:
    client:   1.2 KiB/s rd, 2 op/s rd, 0 op/s wr
    Reference: Ceph Status
    ```
    Debug Mode¶
    Debug mode can be useful when a MON or OSD needs advanced maintenance operations that require the daemon to be stopped. 
    Ceph tools such as ceph-objectstore-tool, ceph-bluestore-tool, or ceph-monstore-tool are commonly used in these scenarios.
    Debug mode will set up the MON or OSD so that these commands can be run.

    Start the debug pod for mon b
    ```
    kubectl rook-ceph debug start rook-ceph-mon-b
    ```
    Stop the debug pod for mon b

    ```
    kubectl rook-ceph debug stop rook-ceph-mon-b
    ```
    """
chunks=[chunk]
sample_promt = """context = '
        Debug mode will set up the MON or OSD so that these commands can be run.

        Start the debug pod for mon b
        ```
        kubectl rook-ceph debug start rook-ceph-mon-b
        ```
        '
        Create 2 different questions and answers for the above context, format like
        Q:
        A:
        """
sample_response ="""
    Q: How do I set up debug mode?
    A: You can set up the debug mode by starting the debug pod for mon b like
    '''
    kubectl rook-ceph debug start rook-ceph-mon-b
    '''
    Q: How do I stop debug mode?
    A: You can stop the debug pods by stopping the debug pod for mon b like
    '''
    kubectl rook-ceph debug stop rook-ceph-mon-b
    '''
    """


for index, chunk in enumerate(chunks):
    #print(f"prompt={chunk}")
    #print("--"*80)
    #continue
    prompt =f"""context = '{chunk}'
    Create different questions and answers using the above context, format like
    Q:
    A:
    Give helpful code snippets in the answers whereever possible.
    """
    prompt_template = mistral7b_4bit.create_prompt(sample_promt,sample_response,prompt)
    try:
        output = mistral7b_4bit.generate_ouputput(prompt_template)
        response =mistral7b_4bit.get_chat_response(output)
        print(response)
        matches = parse_response_out(response)
        with open("qa_data_mistral.txt", "a", encoding='utf-8') as myfile:
            for index, (question, answer) in enumerate(matches):
                out = f"<s>[INST] Source:2212AA {question} [/INST] Source:2212AA {answer} </s>\n"
                myfile.write(out)
            
    except Exception :
        print(traceback.format_exc())

