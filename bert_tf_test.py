# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:27:32 2020

@author: Andrei
"""

import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

if __name__ == '__main__':  
  tokenizer = BertTokenizer.from_pretrained('D:/Lummetry.AI Dropbox/DATA/_allan_data/_ro_bert/20200520')
  model = TFBertModel.from_pretrained('D:/Lummetry.AI Dropbox/DATA/_allan_data/_ro_bert/20200520')
  
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
    pad_to_max_length=True)
  
  t2s = [tokenizer.convert_ids_to_tokens(x) for x in data['input_ids']]
  
  for detokenized in t2s:
    s = ''
    for w in detokenized:
      d = ' ' if w[0] != '#' else ''
      s = s + d + w
    print(s)
  
  np_data = np.array(data['input_ids'])
  np_mask = np.array(data['attention_mask'])
  

  embeds, clf_out = model(np_data, attention_mask=np_mask)
  np_embeds = embeds.numpy()
#  np_py_embeds = np.load('_pytest.npy') # salvat de mine din pytorch
#  print(np.allclose(np_embeds, np_py_embeds, atol=1e-5))
  np_emb, np_clf = model.predict(np_data)
  
  print(np.allclose(np_emb, np_embeds, atol=1e-3))
  
  np_s2 = tokenizer.encode(sents2[0])

  
