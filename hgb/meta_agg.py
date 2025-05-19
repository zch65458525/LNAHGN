import torch
import torch.nn.functional as F
from utils import *
import argparse
from api import API
import re
import xml.etree.ElementTree as ET
from io import StringIO

def get_weight(length):
    file_path = "./cache_data/ACM/meta_agg.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    metapath_blocks = re.findall(r'<metapath>\s*(.*?)\s*</metapath>', content, re.DOTALL)
    name_score_dict = {}
    for block in metapath_blocks:
        xml_str = f"<metapath>{block}</metapath>"
        
        try:
            root = ET.fromstring(xml_str)
            name = root.find('name').text
            score = float(root.find('score').text)
            if len(name)>length+1:
                break
            name_score_dict[name] = score
        except Exception as e:
            print(f"error: {e}")

    print(name_score_dict)
    return name_score_dict
prompt1="""
In a heterogeneous graph neural network using the ACM dataset, there are three types of nodes: Paper (P), Author (A), and Conference (C). 
The goal is to classify the node types of paper nodes. Using the metapath approach, the following metapaths are constructed: {input1}.
Below is an explanation of the practical meanings of these metapaths and their importance scores (a value between 0 and 1) for the node type classification task.
Each metapath should be in this format:
<metapath>
<name>metapath's name</name>
<meaning>metapath's meaning</meaning>
<score>importance score</score>
<reason>reason for the importance score</reason>
</metapath>
"""
def main(args):
    global prompt1
    llm=API()
    g, adjs, init_labels, num_classes, dl, trainval_nid, test_nid,text_attribute = load_dataset(args)
    
    for k in adjs.keys():
        adjs[k].storage._value = None
        adjs[k].storage._value = torch.ones(adjs[k].nnz()) / adjs[k].sum(dim=-1)[adjs[k].storage.row()]
    if args.dataset == 'DBLP':
        tgt_type = 'A'
        node_types = ['A', 'P', 'T', 'V']
        extra_metapath = []
    elif args.dataset == 'ACM':
        tgt_type = 'P'
        node_types = ['P', 'A', 'C']
        extra_metapath = []
    elif args.dataset == 'IMDB':
        tgt_type = 'M'
        node_types = ['M', 'A', 'D', 'K']
        extra_metapath = []
    else:
        assert 0
    extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_hops + 1]

    print(f'Current num hops = {args.num_hops} for feature propagation')
    prop_device = 'cpu'
    store_device = 'cpu'
    if len(extra_metapath):
        max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
    else:
        max_length = args.num_hops + 1
    g = hg_propagate_feat_dgl(g,1, tgt_type, args.num_hops, max_length, extra_metapath, echo=True)
    raw_feats = {}
    keys = list(g.nodes[tgt_type].data.keys())

    prompt1=prompt1.format(input1=str(keys))
    pro=[{"role": "user", "content":prompt1}]
    res=llm.query(pro,num_responses=1)[0]
    f=open('./cache_data/ACM/meta_agg.txt','w')
    f.write(res)
def parse_args(args=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-hops', type=int, default=2,
                        help='number of hops for propagation of raw labels')
    parser.add_argument('--dataset', type=str, default='ACM',
                        choices=['DBLP', 'ACM', 'IMDB', 'Freebase'])
    parser.add_argument('--root', type=str, default='./cache_data/')
    parser.add_argument('--process', type=int, default=0)
    return parser.parse_args(args)
if __name__ == '__main__':
    args = parse_args()
    args.ACM_keep_F = False
    print(args)
    if args.process==1:
        main(args)
    else:
        get_weight()