from chatgpt import Chatgpt
from llm import LLM
from api import API
import os
llm=API()
prompt1="""Please summarize the paper in one sentence using concise language.The title of the paper and the abstract of the paper is {input}."""
prompt2="""The paper may belong to one of the following three fields: database, wireless communication, or data mining. Please determine which field you believe the paper belongs to.The title of the paper and the abstract of the paper is {input}.
show your answer in <field></field>.Only provide the answer, do not generate explanations"""
test="Influence and correlation in social networks  In many online social systems, social ties between users play an important role in dictating their behavior. One of the ways this can happen is through social influence, the phenomenon that the actions of a user can induce his/her friends to behave in a similar way. In systems where social influence exists, ideas, modes of behavior, or new technologies can diffuse through the network like an epidemic. Therefore, identifying and understanding social influence is of tremendous interest from both analysis and design points of view.   This is a difficult task in general, since there are factors such as homophily or unobserved confounding variables that can induce statistical correlation between the actions of friends in a social network. Distinguishing influence from these is essentially the problem of distinguishing correlation from causality, a notoriously hard statistical problem.   In this paper we study this problem systematically. We define fairly general models that replicate the aforementioned sources of social correlation. We then propose two simple tests that can identify influence as a source of social correlation when the time series of user actions is available.   We give a theoretical justification of one of the tests by proving that with high probability it succeeds in ruling out influence in a rather general model of social correlation. We also simulate our tests on a number of examples designed by randomly generating actions of nodes on a real social network (from Flickr) according to one of several models. Simulation results confirm that our test performs well on these data. Finally, we apply them to real tagging data on Flickr, exhibiting that while there is significant social correlation in tagging behavior on this system, this correlation cannot be attributed to social influence."

res=[]
with open(os.path.join('../data/ACM/', 'node.dat'), 'r', encoding='utf-8') as f:
    for line in f:
        th = line.split('\t')
        id,name,_,_=th
        if int(id)>=3025:
            break
        print(name)
        pro=[{"role": "user", "content":prompt1.format(input=name)}]
        res.append(llm.query(pro,num_responses=1,temperature=0.3,top_p=0.95)[0])
        print(res[-1])
        print('\n')
f=open('./cache_data/ACM/acm_summarize.txt','w')
for i in res:
    f.write(i.strip()+'\n')
res=[]
with open(os.path.join('../data/ACM/', 'node.dat'), 'r', encoding='utf-8') as f:
    for line in f:
        th = line.split('\t')
        id,name,_,_=th
        if int(id)>=3025:
            break
        print(name)
        pro=[{"role": "user", "content":prompt2.format(input=name)}]
        res.append(llm.query(pro,num_responses=1,temperature=0.3,top_p=0.95)[0])
        print(res[-1])
        print('\n')
f=open('./cache_data/ACM/persudo_label.txt','w')
for i in res:
    f.write(i.strip()+'\n')