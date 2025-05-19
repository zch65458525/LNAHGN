from openai import OpenAI
import copy
import time

class API():
    def __init__(self):
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key="",
        )

    def query(self,query,num_responses=1,temperature=0.3,top_p=0.7,max_tokens=4000):
        res=[]
        
        
        msg=copy.deepcopy(query)
        #print(msg)
        for i in range(num_responses):
            #print(query)
            #print(msg,'\n')
            response= self.client.chat.completions.create(
                model="deepseek-chat",
                messages=msg,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            #print(response)
            res.append(response.choices[0].message.content)
            
            msg[-1]['content']=msg[-1]['content']#+prompt+"[Generated Plan {}:".format(i+1)+response.choices[0].message.content+"]\n*************\n"
        #print(res)
        #time.sleep(0.2)
        return res