
class PromptGenerator():
   def __init__(self):
       self.system_format = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful,' \
                    ' detailed, and polite answers to the user\'s questions. USER:'
       self.end_format = ' ASSISTANT:'
   def object_prompt_generation(self,objLabel,scene_type):

       if scene_type == '':
           prompt = "What are the typical shape, componets, color of a {}?".format(objLabel)
       else:
           prompt = " In an <{}> typically found in a {}? If so, what are the typical shape," \
                    " components, and color of an <{}> in that context?".format(objLabel,scene_type,objLabel)
           final_prompt = self.system_format+ prompt + self.end_format
       return final_prompt


   def relation_prompt_generation(self,mode,subLabel,relLabel,objLabel, scene_type):
       prompt = ''
       # "In the context of a {}, does the relationship <subject: {}, predicate: {}, object: {}> make sense? If so, could you describe the circumstances or the relationship in detail?"
       if mode == 'SO':
           prompt += "What is the general meaning of the spatial relationship between " \
                    "objects when '<{}>-[PREDICATE]-<{}>'?".format(subLabel,objLabel)
       elif mode == 'SP':
           prompt += "What is the general meaning of the spatial relationship between " \
                    "objects when '<{}>-<{}>-[OBJECT]'?".format(subLabel, relLabel)
       elif mode == 'PO':
           prompt += "What is the general meaning of the spatial relationship between " \
                    "objects when '[SUBJECT]-<{}>-<{}>'? Print only the key sentences, i.e. the clauses below is that".format(relLabel, objLabel)
       elif mode =='SPO':
           # prompt += "What is the general meaning of the spatial relationship between " \
           #           "objects when '<{}>-<{}>-<{}>'? Print only the key sentences".format(subLabel,relLabel, objLabel)

           prompt = "In the context of a {}, is the relationship <subject: {}, predicate: {}, object: {}> possible? " \
                    "If yes, could you please describe the relationship in detail?".format(scene_type,subLabel,
                                                                                                        relLabel,objLabel)
       else:
           prompt += "NO"
       if scene_type != '':
           prompt = "In the context of a {},".format(scene_type) + prompt

       final_prompt = self.system_format + prompt + self.end_format
       return final_prompt

   def room_type_prompt(self,objects):
       obj_list = list(objects.values())
       object_list_word = ", ".join(obj_list)

       prompt = "Given the objects : {}," \
                "witch one room type is most closely associated with them? Choose from: kitchen room, bedroom room, living room, storage room," \
                "dining room, desk room, toilet. Output in one word please.".format(object_list_word)

       final_prompt = self.system_format + prompt + self.end_format
       return final_prompt
