import torch
import torch.nn.functional as F
from utils import *
import argparse
from api import API
import re
import os
import sys
from st import SentenceEncoder
def llm_agg(node_fea,neighbor_fea,cls1,cls2,nhop,text_attribute,write_text=0):
    llm=API()
    encoder=SentenceEncoder('name',device='cpu')
    prompt1="""Currently, experiments are being conducted on the ACM dataset, which contains three types of nodes: papers (P), authors (A), and conferences (C). 
    Select {nhop}-hop neighborhood metapath [{metapath}] for message passing.
    The given central node type is [{class1}], with its text attribute being [{input1}]. The selected {nhop}-hop neighborhood consists of nodes of type [{class2}], with their text attributes being [{input2}]. 
    Please aggregate the neighbor nodes and update a concise yet meaningful representation for the central node. The output format should be <ans>aggregate answer</ans>.Only generate answer itself.answer should be in 1 line.
    """
    res=[]
    encode=[]
    l=len(text_attribute)
    for i in range(len(node_fea)):

        neighbor_fea[i]=random.sample(neighbor_fea[i], min(len(neighbor_fea[i]),15))

        pro=[{"role": "user", "content":prompt1.format(class1=cls1,class2=cls2,nhop=nhop,input1=node_fea[i],input2=neighbor_fea[i],metapath=cls1+cls2)}]
        content=llm.query(pro,num_responses=1)[0]
        ans = re.findall(r'<ans>\s*(.*?)\s*</ans>', content, re.DOTALL)
        if len(ans)==0:
            ans=[content.replace('\n', '').replace('\r', '')]
        else:
            ans[0]=ans[0].replace('\n', '').replace('\r', '')

        text_attribute.append(ans[0])
        res.append(i+l)
        encode.append(encoder.encode(ans[0]))
        
    print(len(text_attribute))
    return torch.stack(encode),torch.tensor(res)
