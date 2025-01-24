import json
import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast
import google.generativeai as palm
from src.fastchat.Prompt import PromptGenerator
from src.fastchat.model.model_FastLlama2 import FastLlama2

import pickle
import openai

openai.api_key = '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'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/baebro/hojun_ws/LMM based 3D Scene Graph Generation(L3DSG)/google_api_file.json"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
palm.configure(api_key='AIzaSyBp41rUm-I0Z-lCnf4Fs4peTMtkrA9idMY')

class KnowledgeModel():
    def __init__(self,config,obj_category,rel_category):
        self.prompt_dataset_dir = '/home/baebro/hojun_ws/2023_02_28_CSGGN_version2/prompt_db_dataset/3dssg_prompt.json'
        self.knowledge_vector_dir = '/home/baebro/hojun_ws/2023_02_28_CSGGN_version2/prompt_db_dataset/3dssg_knowledge_embedding.h5'
        self.obj_category = obj_category
        self.rel_category = rel_category
        self.scene_type_list = config.SCENE_TYPE_LIST

        self.config = config
        self.knowledge_vector_db = h5py.File(self.knowledge_vector_dir, 'r')
        self.LLM_Type = config.KnowledgeModel.LLM_Type

        if self.LLM_Type == 'CB_GPT_1.3B':
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.LLMtokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-1.3B")
            self.LLMmodel = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-1.3B", load_in_8bit=True)
        elif self.LLM_Type == 'Llama2':
            self.LLMmodel = FastLlama2()


        self.objKnowledgeDB = defaultdict(list)
        self.relKnowledgeDB = defaultdict(list)


        if config.KnowledgeModel.EmbeddingModel == 'RoBERTa':
            self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
            self.text_model = RobertaModel.from_pretrained('roberta-base')

            for param in self.text_model.parameters():
                param.requires_grad = False
            # self.BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # self.BERT_model = BertModel.from_pretrained("bert-base-uncased")

        if config.KnowledgeModel.Obj_PRE_Processing:
            db_root = config.KnowledgeModel.KNOWLEDGE_DB_PATH
            self.obj_db = dict()
            for scene_type in self.scene_type_list:
                obj_db_path = os.path.join(db_root,
                                           "obj_knowledge_in_{}.json".format(scene_type))
                with open(obj_db_path, 'r') as f:
                    self.obj_db[scene_type] = json.load(f)

        if config.KnowledgeModel.Rel_PRE_Processing:
            db_root = config.KnowledgeModel.KNOWLEDGE_DB_PATH
            self.rel_db = dict()
            for scene_type in self.scene_type_list:
                rel_db_path = os.path.join(db_root,
                                           "obj_knowledge_in_{}.json".format(scene_type))
                with open(rel_db_path, 'r') as f:
                    self.rel_db[scene_type] = json.load(f)
        self.prompt_generator = PromptGenerator()


            # 관계도 확보 완료되면 위 코드 작성
    # 지식을 저장하는 함수.
    def store_objKnowledge(self,key, knowledge):
        self.objKnowledgeDB[key].append(knowledge)

    def store_relKnowledge(self,key, knowledge):
        self.relKnowledgeDB[key].append(knowledge)

    # 지식을 파일에 저장하는 함수.
    def save_knowledge(self,file_path):
        objfile = os.path.join(file_path,'PriorObjectKnowledge_DB.pkl')
        with open(objfile, 'wb') as file:
            pickle.dump(self.objKnowledgeDB, file)
        print(f"Knowledge saved to {objfile}")

        relfile = os.path.join(file_path, 'PriorRelationKnowledge_DB.pkl')
        with open(relfile, 'wb') as file:
            pickle.dump(self.relKnowledgeDB, file)
        print(f"Knowledge saved to {relfile}")

    def object_prompt_generation(self,objLabel,scene_type):
        if scene_type == '':
            prompt = "What are the typical shape, componets, color of a {}?".format(objLabel)
        else:
            prompt = "### Question:\nWhat are the typical shapes, components, and colors of a {} in a {}?\n\n### Answer:\n".format(objLabel,scene_type)
        return prompt

    def relation_prompt_generation(self,mode,subLabel,relLabel,objLabel, scene_type):
        prompt = ''
        if mode == 'SO':
            prompt += "What is the general meaning of the spatial relationship between " \
                     "objects when '<{}>-[PREDICATE]-<{}>'?".format(subLabel,objLabel)
        elif mode == 'SP':
            prompt += "What is the general meaning of the spatial relationship between " \
                     "objects when '<{}>-<{}>-[OBJECT]'?".format(subLabel, relLabel)
        elif mode == 'PO':
            prompt += "What is the general meaning of the spatial relationship between " \
                     "objects when '[SUBJECT]-<{}>-<{}>'?".format(relLabel, objLabel)
        elif mode =='SPO':
            prompt += "### Question:\nIn the context of a {}, What is the general meaning of the spatial relationship between objects when '<{}-{}-{}>' \n###" \
                      " Answer:".format(scene_type,subLabel, relLabel, objLabel)
            # prompt += "What is the general meaning of the spatial relationship between " \
            #           "objects when '<{}>-<{}>-<{}>'?".format(subLabel,relLabel, objLabel)
        else:
            prompt += "NO"

        return prompt

    def LLM_prompting(self,LLM,prompt,target='object',scene='room',
                      objLabel=None,subLabel=None,relLabel=None):
        if LLM == 'GPT-3':
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=50,
                temperature=0.8
            )
            Textual_priorKnowledge = response.choices[0].text.strip()
        elif LLM == 'CB_GPT_1.3B':
            inputs = self.LLMtokenizer(prompt, return_tensors="pt").to('cuda')

            outputs = self.LLMmodel.generate(**inputs, num_beams=5, pad_token_id=self.LLMtokenizer.eos_token_id,
                                     max_new_tokens=50, early_stopping=True,
                                     no_repeat_ngram_size=2)
            text_output = self.LLMtokenizer.batch_decode(outputs, skip_special_tokens=True)
            Textual_priorKnowledge = text_output[0].split('###')[-1]
            print('Prompt : ', prompt)
            print('CB_GPT_1.3B Result : ', Textual_priorKnowledge)
        elif LLM == 'TEST':
            Textual_priorKnowledge = prompt
        elif LLM == 'PALM':
            defaults = {
                'model': 'models/text-bison-001',
                'temperature': 0.6,
                'candidate_count': 1,
                'top_k': 40,
                'top_p': 0.85,
                'max_output_tokens': 200,
            }
            response = palm.generate_text(

                prompt=prompt)
            Textual_priorKnowledge = response.result
            print('Prompt : ',prompt)
            print('PALM Result : ', Textual_priorKnowledge)

            if Textual_priorKnowledge is None:
                Textual_priorKnowledge = "Unknown"

        elif LLM == 'Llama2':
            if target == 'object':
                if self.config.KnowledgeModel.Obj_PRE_Processing:
                    Textual_priorKnowledge = self.obj_db[scene][scene][objLabel]
                else:
                    Textual_priorKnowledge = self.LLMmodel.generate_stream(prompt)
            elif target == 'relation':
                if self.config.KnowledgeModel.Rel_PRE_Processing:
                    subkey = '{}_{}_{}'.format(subLabel,relLabel,objLabel)
                    Textual_priorKnowledge = self.rel_db[scene][subkey]
                else:
                    if self.config.KnowledgeModel.Rel_using_llm:
                        Textual_priorKnowledge = self.LLMmodel.generate_stream(prompt)['text']
                    else:
                        Textual_priorKnowledge = '{}_{}_{}'.format(subLabel,relLabel,objLabel)
        return Textual_priorKnowledge

    def obj_textEmbedding(self,text,obj,scene):


        if self.config.KnowledgeModel.Obj_Embedding:
            with torch.no_grad():
                find_key = '{}-{}'.format(scene,obj)
                embeddings = torch.from_numpy(self.obj_db[find_key])

        elif self.config.KnowledgeModel.EmbeddingModel == 'GPT-3':
            response = openai.Embedding.create(
                input= text,
                model="text-embedding-ada-002"
            )
            embeddings = response['data'][0]['embedding']

            # size --> always 1536 , type : list,,,

        elif self.config.KnowledgeModel.EmbeddingModel == 'PALM':

            palm.configure(api_key='AIzaSyBfLSu_aWd06xb2lUDeTHsZl9p7aUQusf0')


        elif self.config.KnowledgeModel.EmbeddingModel == 'RoBERTa':

            with torch.no_grad():
                encoded_input = self.tokenizer(text, return_tensors='pt')
                output = self.text_model(**encoded_input)
            embeddings = output.last_hidden_state.mean(1) # tensor, (1,768)

        return embeddings


    def rel_textEmbedding(self,text,sub,rel,obj,scene):
        if self.config.KnowledgeModel.Rel_Embedding:
            with torch.no_grad():
                scene = 'living room'
                find_key = '{}-{}-{}-{}'.format(scene,sub,rel,obj)
                embeddings = torch.from_numpy(self.rel_db[find_key])
                # with torch.no_grad():
                #
                #     encoded_input = self.tokenizer(find_key, return_tensors='pt')
                #     output = self.text_model(**encoded_input)
                # embeddings = output.last_hidden_state.mean(1)  # tensor, (1,768)


        elif self.config.KnowledgeModel.EmbeddingModel == 'GPT-3':
            response = openai.Embedding.create(
                input= text,
                model="text-embedding-ada-002"
            )
            embeddings = response['data'][0]['embedding']
            # size --> always 1536 , type : list,,,

        elif self.config.KnowledgeModel.EmbeddingModel == 'RoBERTa':
            with torch.no_grad():
                encoded_input = self.tokenizer(text, return_tensors='pt')
                output = self.text_model(**encoded_input)
            embeddings = output.last_hidden_state.mean(1) # tensor, (1,768)
        return embeddings


    def object_knowledge_embedding(self,k,logits,obj_scans):
        device = logits.device
        soft_logits = logits.exp()
        # logits = F.softmax(logits,dim=1)
        node_topk_inds = soft_logits.topk(k=k).indices
        obj_k_feats = []
        for i in range(node_topk_inds.shape[0]):
            for referObj_idx in node_topk_inds[i]:
                referObj_type = self.obj_category[referObj_idx]
                scene_type = self.scene_type_list[obj_scans[i]]
                ##Prompt Generation##
                prompt = self.prompt_generator.object_prompt_generation(referObj_type,scene_type)
                # LLM Prompting #
                prior_object_knowledge = self.LLM_prompting(self.LLM_Type, prompt,target='object',scene=scene_type,
                                                            objLabel=referObj_type,subLabel=None,relLabel=None)

                # print('prompt : ', prompt)
                # print('response : ', prior_object_knowledge)

                # knowledge store
                # self.store_objKnowledge(referObj_type, prior_object_knowledge)

                # Textual Data Embedding #
                related_knowledge_feature = self.obj_textEmbedding(prior_object_knowledge,referObj_type,scene_type)

                obj_k_feats.append(related_knowledge_feature)
                ## return type = tensor [ 1, dim ]

        object_prior_knowledge_feature = torch.cat(obj_k_feats, dim=0).to(device)
        # for S-O
        node_top1_label = {}
        node_top1_inds = soft_logits.topk(k=1).indices
        for objIdx,referObj_idx in enumerate(node_top1_inds):
            node_top1_category = self.obj_category[referObj_idx]
            node_top1_label[objIdx] = node_top1_category #+'_{}'.format(objIdx)
                #############################

        return object_prior_knowledge_feature,node_top1_label

    def rel_knowledge_embedding(self,k,edge_index,predicted_obj_label,rel_logits,rel_scans):
        device = rel_logits.device
        # rel_logits = F.softmax(logits,dim=1)
        edges = edge_index.T
        edge_topk_inds = rel_logits.topk(k=k).indices
        rel_k_feats = []
        for i in range(edge_topk_inds.shape[0]):
            for referRel_idx in edge_topk_inds[i]:
                referRel_type = self.rel_category[referRel_idx]
                top_1_subLabel = predicted_obj_label[edges[i][0].item()]
                top_1_objLabel = predicted_obj_label[edges[i][1].item()]

                scene_type = self.scene_type_list[rel_scans[i]]
                ##Prompt Generation##
                prompt = self.prompt_generator.relation_prompt_generation(mode='SPO',subLabel=top_1_subLabel,
                                                         relLabel=referRel_type,objLabel=top_1_objLabel,
                                                         scene_type=scene_type)
                # LLM Prompting #
                prior_rel_knowledge = self.LLM_prompting(self.LLM_Type, prompt,target='relation')

                # print('prompt : ',prompt)
                # print('response : ', prior_rel_knowledge)

                # knowledge store
                # self.store_relKnowledge('{}-{}-{}'.format(top_1_subLabel,referRel_type,top_1_objLabel),
                #                         prior_rel_knowledge)

                # Textual Data Embedding #
                related_knowledge_feature = self.rel_textEmbedding(prior_rel_knowledge,
                                                                   top_1_subLabel,referRel_type,
                                                                   top_1_objLabel,scene_type)
                rel_k_feats.append(related_knowledge_feature)
                ## return type = tensor [ 1, dim ]

        relation_prior_knowledge_feature = torch.cat(rel_k_feats, dim=0).to(device)


                #############################

        return relation_prior_knowledge_feature


    def obj_KnowledgeSearch(self,objectType, sceneType):
        number = random.randint(0, 4)
        key = objectType + '_' + sceneType[0]
        temp_keys = self.knowledge_vector_db.keys()
        vect = self.knowledge_vector_db[f"{key}/{str(number)}"]
        value = self.prompt_db['entity'][key]
        return value[number], np.array(vect)

    def rel_KnowledgeSearch(self,relType, sceneType):
        number = random.randint(0, 4)
        zeros_array = np.zeros((1, 768))
        key = relType + '_' + sceneType[0]

        if relType == 'none':
            # print('predicate is none')
            return relType , zeros_array

        vect = self.knowledge_vector_db[f"{key}/{str(number)}"]

        value = self.prompt_db['predicate'][key]

        return value[number] , np.array(vect)