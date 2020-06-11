# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:27:32 2020

@author: Andrei
"""

import numpy as np
import torch as th
from transformers import BertModel, BertTokenizer, EncoderDecoderModel

if __name__ == '__main__':  
  tokenizer = BertTokenizer.from_pretrained("D:/Lummetry.AI Dropbox/DATA/_allan_data/_ro_bert/20200520")
  model = BertModel.from_pretrained("D:/Lummetry.AI Dropbox/DATA/_allan_data/_ro_bert/20200520")
  model.eval()
  
  sents1 = [
      "Ana are foooarte multe mere si mai are si super leafa marisoara",
      "Cat e salariul la EY?",
      ]
  
  sents2 = [
      "Cat aveti la EY leafa in Bucuresti?",
      ]
  
  
  data = tokenizer.batch_encode_plus(
    sents1, 
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=512)
  
  t2s = [tokenizer.convert_ids_to_tokens(x) for x in data['input_ids']]
  
  for detokenized in t2s:
    s = ''
    for w in detokenized:
      d = ' ' if w[0] != '#' else ''
      s = s + d + w
    print(s)
  
  np_data = np.array(data['input_ids'])
  np_mask = np.array(data['attention_mask'])
  
  th_data = th.tensor(np_data).long()
  th_mask = th.tensor(np_mask)
  with th.no_grad():
    embeds, pooling = model(th_data, attention_mask=th_mask)
  np_embeds = embeds.numpy()    
  np.save('_pytest.npy', np_embeds)
  
  th_s2 =tokenizer.encode(sents2[0], return_tensors='pt')
  
#  ed = EncoderDecoderModel.from_encoder_decoder_pretrained('ro_bert','ro_bert')
#  ed.generate(th_s2, max_length=50, decoder_start_token_id=model.config.decoder.pad_token_id)
  
    
  
