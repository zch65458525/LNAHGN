import os
import sys
import gc
import random
from collections import defaultdict

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

import numpy as np
import scipy.sparse as spa
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.append('../data')
from data_loader import data_loader

import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

from st import SentenceEncoder

device=torch.device("cuda:0")
def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluator(gt, pred):
    gt = gt.cpu().squeeze()
    pred = pred.cpu().squeeze()
    return f1_score(gt, pred, average='micro'), f1_score(gt, pred, average='macro')


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
def message_func(src_feature_name):

    def message_func(edges):

        if len(src_feature_name)==2:
            print('edge',edges.src)
            return {'text':edges.src['text']}
        else:
            return {'text':edges.src['text']}
    return message_func

    
def reduce_func(dst_feature_name,text_attribute,metapath):

    def reducer(nodes):
        node_text_feature=[text_attribute[id] for id in nodes.data['text']]
        
        decoded_sentences = [decode_sentence(row, text_attribute) for row in nodes.mailbox['text']]
        
        from nei_agg import llm_agg
        print(metapath)
        encode,text_index=llm_agg(node_text_feature,decoded_sentences,metapath[0],metapath[1:],len(metapath[1:]),text_attribute)
        encode=encode.cpu()
        text_index=text_index.cpu()
        return {metapath: encode,metapath+"_text":text_index}
    return reducer
def decode_sentence(ids, word_dictionary):

    ids=ids.tolist()
    words=[word_dictionary[i] for i in ids]
    return words
def get_text_attribute(text_attribute,path,process=0):
    if process==1:
        pass
    else:
        f=open(path,'r')
        i=0
        for line in f.readlines():
            text_attribute[i]=line
            i+=1
def split_into_pairs(s):
    if len(s) <= 1:
        return [s]
    return [f"{s[i]}-{s[i+1]}" for i in range(len(s)-1)]
    
def hg_propagate_feat_dgl(g,text_attribute, tgt_type, num_hops, max_length, extra_metapath,dataset,weight,init_labels, echo=False):
    #print(text_attribute)
    print(weight)
    meta_path={ntype:[ntype] for ntype in g.ntypes}
    encoder=SentenceEncoder('name',device='cpu')
    for ntype in g.ntypes:
        if os.path.exists(f'./cache_data/{dataset}/{ntype}.pt'):
            g.nodes[ntype].data[f'{ntype}']=torch.load(f'./cache_data/{dataset}/{ntype}.pt')
        else:
            decoded_sentences = decode_sentence(g.nodes[ntype].data["text"], text_attribute)
            encode=[]
            for row in decoded_sentences:
                encode.append(encoder.encode(row))

            encode=torch.stack(encode)
            g.nodes[ntype].data[ntype]=encode
            torch.save(g.nodes[ntype].data[ntype],f'./cache_data/{dataset}/{ntype}.pt')
    
    for hop in range(1, max_length):
        reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)
            for k in list(meta_path[stype]):
                if k=='text':
                    continue
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
                      or (hop > num_hops and k not in reserve_heads):
                        continue
                    if echo: print(k, etype, current_dst_name)
                    g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)
                    meta_path[dtype].append(current_dst_name)

        for ntype in g.ntypes:
            removes = []
            
               
            
            for k in meta_path[ntype]:
                if ntype == tgt_type: 
                    if weight[k]<0.6:
                        removes.append(k)
                        #continue
                if k=='text':
                    continue
                if len(k) <= hop:
                    removes.append(k)
            print(removes)
            for k in removes:
                #print(meta_path[ntype],k)
                meta_path[ntype].remove(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo: print(f'-- hop={hop} ---')
        for ntype in g.ntypes:
            for k in meta_path[ntype]:
                if k=='text':
                    continue
                print(f'{ntype} {k}')
        if echo: print(f'------\n')
    text_attribute=text_attribute.tolist()
    for ntype in g.ntypes:
        if '_ID' not in g.nodes[ntype].data:
    
            g.nodes[ntype].data['_ID'] = torch.arange(g.number_of_nodes(ntype))
    
    for m in meta_path[tgt_type]:
        
        if len(m)==1:
            continue

        print("metapath:",m)
        sp=m
        if os.path.exists(f'./cache_data/{dataset}/{sp}.txt'):
            f=open(f'./cache_data/{dataset}/{sp}.txt','r')
            lines = [line.strip() for line in f]
            index=torch.tensor(range(len(text_attribute),len(text_attribute)+len(lines)))
            text_attribute.extend(lines)
            g.nodes[sp[0]].data[f'{sp}_text']=index
            g.nodes[sp[0]].data[f'{sp}']=torch.load(f'./cache_data/{dataset}/{sp}.pt')
            print(g.nodes[sp[0]].data.keys())
        else:
            stype=sp[1]
            dtype=sp[0]
            metapath=split_into_pairs(m)
            tmp=None
            metapath.reverse()
            
            for i in metapath:
                if tmp!=None:
                    tmp=g.adj(etype=(i[0], i, i[-1]))@tmp
                else:
                    tmp=g.adj(etype=(i[0], i, i[-1]))
                if tmp.shape[0]==tmp.shape[1]:
                    tmp=tmp.to_dense()
                    tmp=tmp-torch.diag(tmp.diagonal())
            tmp=tmp.to_dense()
            u, v = tmp.nonzero(as_tuple=True)
            print(dtype,sp[-1])
            if len(u)==0:
                continue
            print(torch.max(u),torch.max(v))
            if dtype==sp[-1]:
                meta_g=dgl.graph((u, v), num_nodes=g.number_of_nodes(dtype))
            else:
                meta_g=dgl.heterograph({
                    (sp[-1], sp[-1]+"-"+dtype, dtype): (v,u)
                },
                num_nodes_dict={sp[-1]: g.number_of_nodes(sp[-1]), dtype: g.number_of_nodes(dtype)})
            print(meta_g)
            print(g)
            
            if dtype==sp[-1]:
                
                for feat_name in g.nodes[dtype].data:
                    meta_g.ndata[feat_name] = g.nodes[dtype].data[feat_name].clone()
                meta_g.update_all(
                    message_func(sp),
                    reduce_func(current_dst_name,text_attribute,sp),
                )
                
                
                g.nodes[dtype].data[sp] = meta_g.ndata[sp]
                g.nodes[dtype].data[f'{sp}_text'] = meta_g.ndata[f'{sp}_text']
            else:
                for ntype in meta_g.ntypes:
                    for feat_name in g.nodes[ntype].data:
                        meta_g.nodes[ntype].data[feat_name] = g.nodes[ntype].data[feat_name].clone()
                meta_g.update_all(
                    message_func(sp),
                    reduce_func(current_dst_name,text_attribute,sp),
                )
                
                
                g.nodes[dtype].data[sp] = meta_g.nodes[dtype].data[sp]
                g.nodes[dtype].data[f'{sp}_text'] = meta_g.nodes[dtype].data[f'{sp}_text']
            
            f=open(f'./cache_data/{dataset}/{sp}.txt','w')
            rows=[]
            for row in g.nodes[dtype].data[f'{sp}_text']:
                if row.item()==0:
                    text_attribute.append('None')
                    rows.append(len(text_attribute)-1)
                else:
                    rows.append(row.item())
            decoded_sentences = decode_sentence(torch.tensor(rows), text_attribute)
            for i in range(len(decoded_sentences)):
                f.write(decoded_sentences[i]+"\n")
                    
            torch.save(g.nodes[dtype].data[sp],f'./cache_data/{dataset}/{sp}.pt')
            f.close()
        
    
    for ntype in g.ntypes:
        feature_names = list(g.nodes[ntype].data.keys())
        text_features = [name for name in feature_names if 'text' in name or '_ID' in name]
        for feat_name in text_features:
            del g.nodes[ntype].data[feat_name]
    
    return g



def train(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator,labels, mask=None, scalar=None):
    #print(labels)
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []

    for batch in train_loader:
        # batch = batch.to(device)
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch, batch_feats, batch_labels_feats, batch_mask,labels)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att = model(batch, batch_feats, batch_labels_feats, batch_mask,labels)
            
            loss_train = loss_fcn(output_att, batch_y)
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc


def load_dataset(args):
    dl = data_loader(f'{args.root}/{args.dataset}')

    # use one-hot index vectors for nods with no attributes
    # === feats ===
    features_list = []
    text_list=[]
    
    text_attribute={}
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        tx=dl.nodes['text_index'][i]
        
        if th is None:
            features_list.append(torch.eye(dl.nodes['count'][i]))
            text_list.append(torch.tensor(tx))
            
        else:
            features_list.append(torch.FloatTensor(th))
            text_list.append(torch.tensor(tx))
            

    idx_shift = np.zeros(len(dl.nodes['count'])+1, dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        idx_shift[i+1] = idx_shift[i] + dl.nodes['count'][i]

    # === labels ===
    num_classes = dl.labels_train['num_classes']
    init_labels = np.zeros((dl.nodes['count'][0], num_classes), dtype=int)
    #print(len(init_labels))
    vector=np.arange(len(init_labels))
    seed=np.random.seed(114514)
    indices = np.random.permutation(len(init_labels))
    selected_indices = indices[:int(0.2*len(init_labels))]
    remaining_indices = indices[int(0.2*len(init_labels)):]
    
    trainval_nid = np.nonzero(dl.labels_train['mask'])[0]
    
    test_nid = np.nonzero(dl.labels_test['mask'])[0]
    #print(trainval_nid,test_nid)
    

    init_labels[trainval_nid] = dl.labels_train['data'][trainval_nid]
    init_labels[test_nid] = dl.labels_test['data'][test_nid]
    
    trainval_nid = vector[selected_indices]
    test_nid = vector[remaining_indices]
    print(len(trainval_nid),len(test_nid))
    print(np.any(np.isin(trainval_nid, test_nid)))
    if args.dataset != 'IMDB':
        init_labels = init_labels.argmax(axis=1)
    init_labels = torch.LongTensor(init_labels)

    adjs = [] if args.dataset != 'Freebase' else {}
    for i, (k, v) in enumerate(dl.links['data'].items()):
        v = v.tocoo()
        src_type_idx = np.where(idx_shift > v.col[0])[0][0] - 1
        dst_type_idx = np.where(idx_shift > v.row[0])[0][0] - 1
        row = v.row - idx_shift[dst_type_idx]
        col = v.col - idx_shift[src_type_idx]
        sparse_sizes = (dl.nodes['count'][dst_type_idx], dl.nodes['count'][src_type_idx])
        adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=sparse_sizes)
        if args.dataset == 'Freebase':
            name = f'{dst_type_idx}{src_type_idx}'
            assert name not in adjs
            adjs[name] = adj
        else:
            adjs.append(adj)
            print(adj)

    if args.dataset == 'DBLP':
        # A* --- P --- T
        #        |
        #        V
        # author: [4057, 334]
        # paper : [14328, 4231]
        # term  : [7723, 50]
        # venue(conference) : None
        A, P, T, V = features_list
        AP, PA, PT, PV, TP, VP = adjs

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('P', 'P-A', 'A'),
            ('A', 'A-P', 'P'),
            ('T', 'T-P', 'P'),
            ('V', 'V-P', 'P'),
            ('P', 'P-T', 'T'),
            ('P', 'P-V', 'V'),
        ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)
        g = dgl.heterograph(new_edges)


        # g.ndata['feat']['A'] = A # not work
        g.nodes['A'].data['A'] = A
        g.nodes['P'].data['P'] = P
        g.nodes['T'].data['T'] = T
        g.nodes['V'].data['V'] = V
    elif args.dataset == 'IMDB':
        # A --- M* --- D
        #       |
        #       K
        # movie    : [4932, 3489]
        # director : [2393, 3341]
        # actor    : [6124, 3341]
        # keywords : None
        M, D, A, K = features_list
        MD, DM, MA, AM, MK, KM = adjs
        assert torch.all(DM.storage.col() == MD.t().storage.col())
        assert torch.all(AM.storage.col() == MA.t().storage.col())
        assert torch.all(KM.storage.col() == MK.t().storage.col())

        assert torch.all(MD.storage.rowcount() == 1) # each movie has single director

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('D', 'D-M', 'M'),
            ('M', 'M-D', 'D'),
            ('A', 'A-M', 'M'),
            ('M', 'M-A', 'A'),
            ('K', 'K-M', 'M'),
            ('M', 'M-K', 'K'),
        ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)
        g = dgl.heterograph(new_edges)

        g.nodes['M'].data['M'] = M
        g.nodes['D'].data['D'] = D
        g.nodes['A'].data['A'] = A
        if args.num_hops > 2 or args.two_layer:
            g.nodes['K'].data['K'] = K
    elif args.dataset == 'ACM':
        # A --- P* --- C
        #       |
        #       K
        # paper     : [3025, 1902]
        # author    : [5959, 1902]
        # conference: [56, 1902]
        # field     : None
        P, A, C, K = features_list
        P_,A_,C_,K_=text_list
        PP, PP_r, PA, AP, PC, CP, PK, KP = adjs
        row, col = torch.where(P)
        assert torch.all(row == PK.storage.row()) and torch.all(col == PK.storage.col())
        assert torch.all(AP.matmul(PK).to_dense() == A)
        assert torch.all(CP.matmul(PK).to_dense() == C)

        assert torch.all(PA.storage.col() == AP.t().storage.col())
        assert torch.all(PC.storage.col() == CP.t().storage.col())
        assert torch.all(PK.storage.col() == KP.t().storage.col())

        row0, col0, _ = PP.coo()
        row1, col1, _ = PP_r.coo()
        PP = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=PP.sparse_sizes())
        PP = PP.coalesce()
        PP = PP.set_diag()
        adjs = [PP] + adjs[2:]

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('P', 'P-P', 'P'),
            ('A', 'A-P', 'P'),
            ('P', 'P-A', 'A'),
            ('C', 'C-P', 'P'),
            ('P', 'P-C', 'C'),
        ]
        if args.ACM_keep_F:
            etypes += [
                ('K', 'K-P', 'P'),
                ('P', 'P-K', 'K'),
            ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)

        g = dgl.heterograph(new_edges)

        g.nodes['P'].data['P'] = P # [3025, 1902]
        g.nodes['A'].data['A'] = A # [5959, 1902]
        g.nodes['C'].data['C'] = C # [56, 1902]
        f=open('./cache_data/ACM/acm_summarize.txt','r')
        f1=open('./cache_data/ACM/persudo_label.txt','r')
        g.nodes['P'].data['text'] = P_ # [3025, 1902]
        g.nodes['A'].data['text'] = A_ # [5959, 1902]
        g.nodes['C'].data['text'] = C_ # [56, 1902]
        text_attribute=dl.nodes['text']#torch.cat([P_,A_,C_],dim=0)#dl.nodes['text']
        tmp=f.readlines()
        tmp1=f1.readlines()
        for i,t in enumerate(tmp):
            tmp[i]=t+tmp1[i]
        text_attribute=np.concatenate((tmp,text_attribute[1],text_attribute[2], text_attribute[3]))

        if args.ACM_keep_F:
            g.nodes['K'].data['K'] = K # [1902, 1902]
    else:
        assert 0

    if args.dataset == 'DBLP':
        adjs = {'AP': AP, 'PA': PA, 'PT': PT, 'PV': PV, 'TP': TP, 'VP': VP}
    elif args.dataset == 'ACM':
        adjs = {'PP': PP, 'PA': PA, 'AP': AP, 'PC': PC, 'CP': CP}
    elif args.dataset == 'IMDB':
        adjs = {'MD': MD, 'DM': DM, 'MA': MA, 'AM': AM, 'MK': MK, 'KM': KM}
    else:
        assert 0

    return g, adjs, init_labels, num_classes, dl, trainval_nid, test_nid,text_attribute


class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
