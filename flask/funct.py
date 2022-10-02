# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 19:13:48 2022

@author: 
"""

import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

import pandas as pd
import numpy as np
import PyPDF2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import spacy
from collections import Counter
from io import StringIO
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import re
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from string import punctuation
import string
from bs4 import BeautifulSoup
import unicodedata
import contractions
from nltk.corpus import webtext
from nltk.probability import FreqDist
 

def WordFreq(text,nlp):       
    filtered_sentence =[] 
    doc=nlp(text)
    for token in doc:
        if token.is_stop == False: 
          filtered_sentence.append(token.text)
    filtered_sentence= ' '.join(filtered_sentence)
    newText=''.join([c for c in filtered_sentence if c not in string.punctuation])
    
    token=re.findall('\w+', newText)
    data_analysis = nltk.FreqDist(token)     
    # Let's take the specific words only if their frequency is greater than 3.
    filter_words = dict([(m, n) for m, n in data_analysis.items() if len(m) > 0])
    fig = plt.figure(figsize = (10,10))
  
    #for key in sorted(filter_words):
     #   print("%s: %s" % (key, filter_words[key]))   
    data_analysis.plot(10, cumulative=False)
    fig.savefig('static/wordfreq.png')

    return filter_words

#nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
#a=remove_stopwords('I love going to school love.',nlp)

def expand_contractions(text):
    expanded_words = [] 
    for word in text.split():
       expanded_words.append(contractions.fix(word)) 
    return ' '.join(expanded_words)

def remove_html_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()

def standardize_accented_chars(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def clean_body(text):
    text=text.replace('\\n', ' ')
    text=text.replace('/s', ' ')
    text=text.replace('\t', ' ')
    text=text.replace('\n', ' ')
    newText = text.lower()
    newText = re.sub('[^\w\s\.]',' ',newText)
    
    #remove special char
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    newText = re.sub(pat,' ', newText)
    #remove URL
    newText = re.sub(r'https?:\S*',' ', newText)
    #remove Hashtag
    newText = re.sub(r'@\S*',' ', newText)
    newText = re.sub(r'#\S*',' ', newText)
    #remove contraction
    newText=expand_contractions(newText)
    #remove Html
    newText=remove_html_tags(newText)
    #remove digit
   # pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
   # newText= re.sub(pattern, ' ', newText)
    #remove punctution except of full stop
    pattern = "[!\"#$%&'()*+,\-/:;<=>?@[\]^_`{|}~]"
    newText= re.sub(pattern, ' ', newText)

    #newText=''.join([c for c in newText if c not in string.punctuation])
    #newText =' '.join(newText.split())
    # tokens = [w for w in newText.split() if not w in STOP_WORDS]
    # long_words=[]
    # for i in tokens:
    #     if len(i)>=3:
    #         long_words.append(i)                                          
    return newText 
    # (" ".join(long_words)).strip()

def summary1(file):
    #read pdf
    file_path=file
    filereader=PyPDF2.PdfFileReader(file_path, 'rb')
    
    #replace space ,tab etc
    pge_count=filereader.getNumPages()
    count=0
    text=[]
    while count<pge_count:
      obj=filereader.getPage(count)
      count+=1
      text.append(obj.extractText())
    text=str(text)
    
    print('Information:\n File Name : ',file_path,'\n No of pages : ',count)
    #text
    
    #Clean text
    clean_text=clean_body(text)
    
    #Summary1
    from gensim.summarization import summarize
    s1=summarize(clean_text,split=False)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    FreqWord1=WordFreq(s1,nlp)
    #SFreqWord1=sorted(FreqWord1, key=FreqWord1.get, reverse=True)[:5]
    SFreqWord1=dict(Counter(FreqWord1).most_common(10))
    return s1,SFreqWord1
    


#nlp= spacy.load("en_core_web_sm")
# doc=nlp(clean_text)

# a='I love going to school.'
# stop_words= list(STOP_WORDS)
# word_freq={}
# for word in a:
#     if word.text.lower() not in stop_words:
#       if word.text.lower() not in punctuation:
#         if word.text not in word_freq.keys():
#           word_freq[word.text]= 1
#         else:
#           word_freq[word.text]+= 1 
# print(word_freq)

# x=(word_freq.values())
# a=list(x)
# a.sort()
# max_freq=a[-1]
# max_freq

# sent_score={}
# sent_tokens=[sent for sent in doc.sents]
# print(sent_tokens)

# for sent in sent_tokens:
#    for word in sent:
#      if word.text.lower() in word_freq.keys():
#        if sent not in sent_score.keys():
#          sent_score[sent]=word_freq[word.text.lower()]
#        else:
#          sent_score[sent]+= word_freq[word.text.lower()] 
# print(sent_score)

# from heapq import nlargest
# len(sent_score) *0.3

# summary=nlargest(n=13,iterable=sent_score,key=sent_score.get) 
# final_summary=[word.text for word in summary]
# final_summary

#f1=[]
#for sub in final_summary:
#  f1.append(re.sub('n','',sub))
#f2=" ".join(f1)
#f2



#Summary2
# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
# model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

# inputs = tokenizer([clean_text], truncation=True, return_tensors='pt')
# # # Generate Summary
# summary_ids = model.generate(inputs['input_ids'],min_length=0)
# summarized_text = ([tokenizer.decode(g) for g in summary_ids])
# summarized_text[0]

def gen_title(a):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer_t = AutoTokenizer.from_pretrained("AryanLala/autonlp-Scientific_Title_Generator-34558227")
    model_t = AutoModelForSeq2SeqLM.from_pretrained("AryanLala/autonlp-Scientific_Title_Generator-34558227")
            
    inputs_t = tokenizer_t(a, truncation=True, return_tensors='pt')
            
     
        # # # Generate title
    title_ids = model_t.generate(inputs_t['input_ids'],min_length=0)
    summarized_title = ([tokenizer_t.decode(g) for g in title_ids])
    summarized_title[0]
    title = summarized_title[0].replace('<pad>', '')
    return title.replace('</s>', '')  

#Summary3

def summary2(file):
    #read pdf
    file_path=file
    filereader=PyPDF2.PdfFileReader(file_path, 'rb')
    
    #replace space ,tab etc
    pge_count=filereader.getNumPages()
    count=0
    text=[]
    while count<pge_count:
      obj=filereader.getPage(count)
      count+=1
      text.append(obj.extractText())
    text=str(text)
    
   # print('Information:\n File Name : ',file_path,'\n No of pages : ',count)
    #text
    
    #Clean text
    clean_text=clean_body(text)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum")
    
    inputs = tokenizer([clean_text], truncation=True, return_tensors='pt')
    # # Generate Summary
    summary_ids = model.generate(inputs['input_ids'],min_length=0)
    summarized_text2 = ([tokenizer.decode(g) for g in summary_ids])
    
    clean_text=re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', clean_text)
    clean_text = re.sub(r"\d+", "", clean_text) 
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    FreqWord2=WordFreq(clean_text,nlp)
    #SFreqWord1=sorted(FreqWord1, key=FreqWord1.get, reverse=True)[:5]
    SFreqWord2=dict(Counter(FreqWord2).most_common(10))
    
    from wordcloud import WordCloud
    wc = WordCloud()
    img = wc.generate_from_text(' '.join(SFreqWord2))
    img.to_file('static/wordcloud.png') # example of something you can
        
    values = list(SFreqWord2.keys())
    value=' '.join(values)
    res = value.replace(' ', ',')
    res = res.replace('sss', '')
    
  
    c = gen_title(summarized_text2[0])
    return summarized_text2[0],res,c
    

#a,b,c=summary2('3.pdf')



