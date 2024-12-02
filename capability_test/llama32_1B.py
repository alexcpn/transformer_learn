from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
import logging as log
import sys

# does not fit into 8 GB RAN

log.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                level=log.DEBUG, datefmt='%I:%M:%S',
                handlers=[
                    log.StreamHandler()
                ])

def create_prompt(question,system_message=None):
        """
            https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/
        """
        if not system_message:
            system_message = "You are a helpful assistant.Please answer the question if it is possible" #this is the default one
       
        prompt_template=f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
        #print(prompt_template)
        return prompt_template
#-------------------------------

def gpt_gen(tokenizer, model, prompt,device):

    encoded_input = tokenizer(prompt, truncation=True,  return_tensors="pt")
    outputs = model.generate(input_ids = encoded_input.input_ids.to(device),attention_mask=encoded_input.attention_mask.to(device),max_new_tokens=2000)
    output = tokenizer.batch_decode(outputs,  skip_special_tokens=True)
    formatted_output = "\n".join(output)
    log.info(10 * '-') 
    log.info(f"Response {formatted_output}") 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    deviceid = torch.cuda.current_device()
    log.info(f"Gpu device {torch.cuda.get_device_name(deviceid)}")
    
bf16 = False
fp16 = True
model_name_long ="meta-llama/Llama-3.2-1B-Instruct" # loads in 6 GB GPU RAM directly


major, _ = torch.cuda.get_device_capability()
if major >= 8:
    print("=" * 80)
    log.info("Your GPU supports bfloat16: accelerate training with bf16=True")
    print("=" * 80)
    bf16 = True
    fp16 = False

# Load the entire model on the GPU 0
device_map = {"": 0} # lets load on the next
#device = torch.device('cuda:0')

# Load base model
if bf16:
    torch_dtype=torch.bfloat16
else:
    torch_dtype=torch.float16

 # Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = torch_dtype
compute_dtype = torch_dtype
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Quantization type (fp4 or nf4)
bnb_8bit_quant_type = "nf8"

bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type=bnb_8bit_quant_type,
        bnb_8bit_compute_dtype=compute_dtype,
        bnb_8bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=True,
)

# This works, this is training the qunatised model

tokenizer = AutoTokenizer.from_pretrained(model_name_long)
model = AutoModelForCausalLM.from_pretrained(
    model_name_long,
    torch_dtype=torch_dtype,
    device_map=device_map,
    #quantization_config=bnb_config,
    trust_remote_code=True

)


# prompt = "translate English to French the following text 'what do the 3 dots mean in math?' "  # «Que signifient les 3 points en mathématiques?»
# prompt_template = create_prompt(prompt,system_message= "You are a helpful AI translator.Please translate if it is possible")
# gpt_gen(tokenizer, model, prompt_template,device)

prompt ="""# https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.452-16-201507-S!!PDF-E.pdf
# [n.7]ITU-R P.452: Prediction procedure for the evaluation of interference between stations on
# the surface of the Earth at frequencies above about 0.1 GHz (P.452-16 (07/2015)),
# available at https://www.itu.int/rec/R-REC-P.452-16-201507-I/en

assert  not (ap_clutter == 0)
"""
prompt_template = create_prompt(prompt,"extract all methods and classes of the source code provided along with line numbers")
gpt_gen(tokenizer, model, prompt_template,device)

sys.exit(0)

prompt ="""
Description:
Two drains where incorrectly detected on different failure domains caused rook to create two drain events protecting different nodes with pdbs and so causing softlock as now all nodes had pbs blocking them to be drained.

Events that caused the issue:

node-3 was drained fine
osd-0 on node-3 did not get healthy state in k8s api but it joined ceph cluster and was able to get all pgs active+clean
rook selects to stop drain protection as cluster is active+clean
pdbs are restored to default
osd-0 is still detected as down by rook
osd-0 becames active
osd-2 is drained (as default pdb allows one pod to be down and osd-0 is not up) (node-1)
rook does the checking of pgs for osd-0 and finds non-active pgs (because osd-2 caused them, but rook do not know that, as same pgs are also on other osds)
rook creates pdbs to block other nodes than node-3
rook detects osd-2 to be down.
rook creates pdbs to protect osd-2
softlock: all nodes are blocked and osd-2 is down and it will not come up until other osds on that node are drained, that will never happen because pdbs block it
Issues:

getting k8s and ceph status is not atomic. So those do not fully reflect the reality.
Two drains can block each others.
Possible solutions:
When setting pdbs because drain, it might be good to check if this node already has pdb blocking draining from it. Solution could be to add the pdb for other nodes and then clean the pdb matching to this node. This would clear this case but I am not sure if there are other cases where this could cause issues.
This might need check from k8s api to see if this failure domain is the only one that has failed osds before clearing the pdb for it. So that we do not accidentally cause multiple failures at same time.

Logs from the event:
2024-09-12 23:03:14.956779 I | clusterdisruption-controller: osd "rook-ceph-osd-0" is down but no node drain is detected
2024-09-12 23:03:15.285896 I | clusterdisruption-controller: osd is down in failure domain "node-3" is down for the last 21.94 minutes, but pgs are active+clean
2024-09-12 23:03:15.556659 I | clusterdisruption-controller: all PGs are active+clean. Restoring default OSD pdb settings
2024-09-12 23:03:15.556674 I | clusterdisruption-controller: creating the default pdb "rook-ceph-osd" with maxUnavailable=1 for all osd
2024-09-12 23:03:15.559175 I | clusterdisruption-controller: deleting temporary blocking pdb with "rook-ceph-osd-host-node-1" with maxUnavailable=0 for "host" failure domain "node-1"
2024-09-12 23:03:15.561094 I | clusterdisruption-controller: deleting temporary blocking pdb with "rook-ceph-osd-host-node-2" with maxUnavailable=0 for "host" failure domain "node-2"
2024-09-12 23:03:15.563269 E | clusterdisruption-controller: failed to update configMap "rook-ceph-pdbstatemap" in cluster "rook-ceph/rook-ceph-cluster": Operation cannot be fulfilled on configmaps "rook-ceph-pdbstatemap": the object has been modified; please apply your changes to the latest version and try again
2024-09-12 23:03:18.125054 I | clusterdisruption-controller: osd "rook-ceph-osd-0" is down but no node drain is detected
2024-09-12 23:03:18.461688 I | clusterdisruption-controller: osd is down in failure domain "node-3" and pgs are not active+clean. pg health: "cluster is not fully clean. PGs: [{StateName:active+clean Count:76} {StateName:peering Count:31} {StateName:active+undersized+degraded Count:22}]"
2024-09-12 23:03:20.899793 I | op-osd: updating OSD 2 on node "node-1"
2024-09-12 23:03:20.909820 I | ceph-cluster-controller: hot-plug cm watcher: running orchestration for namespace "rook-ceph" after device change
2024-09-12 23:03:20.915974 I | clusterdisruption-controller: creating temporary blocking pdb "rook-ceph-osd-host-node-1" with maxUnavailable=0 for "host" failure domain "node-1"
2024-09-12 23:03:20.920432 I | clusterdisruption-controller: creating temporary blocking pdb "rook-ceph-osd-host-node-2" with maxUnavailable=0 for "host" failure domain "node-2"
2024-09-12 23:03:20.920715 I | op-osd: updating OSD 5 on node "node-1"
2024-09-12 23:03:20.926573 I | clusterdisruption-controller: deleting the default pdb "rook-ceph-osd" with maxUnavailable=1 for all osd
2024-09-12 23:03:20.952527 E | clusterdisruption-controller: failed to update configMap "rook-ceph-pdbstatemap" in cluster "rook-ceph/rook-ceph-cluster": Operation cannot be fulfilled on configmaps "rook-ceph-pdbstatemap": the object has been modified; please apply your changes to the latest version and try again
2024-09-12 23:03:20.956665 I | clusterdisruption-controller: osd "rook-ceph-osd-2" is down and a possible node drain is detected
2024-09-12 23:03:20.982523 I | op-osd: waiting... 2 of 2 OSD prepare jobs have finished processing and 4 of 6 OSDs have been updated
2024-09-12 23:03:20.982537 I | op-osd: restarting watcher for OSD provisioning status ConfigMaps. the watcher closed the channel
2024-09-12 23:03:21.254436 E | ceph-spec: failed to update cluster condition to {Type:Ready Status:True Reason:ClusterCreated Message:Cluster created successfully LastHeartbeatTime:2024-09-12 23:03:21.248367805 +0000 UTC m=+1343.926592861 LastTransitionTime:2024-09-04 17:17:51 +0000 UTC}. failed to update object "rook-ceph/rook-ceph-cluster" status: Operation cannot be fulfilled on cephclusters.ceph.rook.io "rook-ceph-cluster": the object has been modified; please apply your changes to the latest version and try again
2024-09-12 23:03:21.293972 I | clusterdisruption-controller: osd is down in failure domain "node-1" and pgs are not active+clean. pg health: "cluster is not fully clean. PGs: [{StateName:active+clean Count:76} {StateName:active+undersized+degraded Count:53}]"
2024-09-12 23:03:22.673733 I | op-osd: OSD 1 is not ok-to-stop. will try updating it again later
2024-09-12 23:03:23.314053 I | clusterdisruption-controller: creating temporary blocking pdb "rook-ceph-osd-host-node-3" with maxUnavailable=0 for "host" failure domain "node-3
"""
prompt_template = create_prompt(prompt,"You are an expert SRE engineer. Anylyze the issue and provide a feeback")
gpt_gen(tokenizer, model, prompt_template,device)