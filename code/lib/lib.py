import numpy as np
import math
import pandas as pd
from functools import partial
import string
removecharlist=string.punctuation+(''.join(np.arange(10).astype(str))) ##"!?#.,\'()`%+&:-;"+'"'
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words = [x.translate({ord(c):None for c in removecharlist}) for x in stop_words]
from nltk.stem.porter import PorterStemmer


def remove_punc(wordlist):

#     print(wordlist)
    extra_wordlist=[]
    for wit,word in enumerate(wordlist):
        word=word.lower()
        word=word.replace('′',"'")
        word=word.replace('`',"'")
        word=word.replace('’',"'")
        if set(word).pop()=='$':
            word='money'
        if word[:-2]=="'s":
            word=word[:-2]
        if "n't"==word[:-3]:
            word=word[:-3]
            extra_wordlist.append('not')
        if "'ve"==word[:-3]:
            wordlist[wit]=word[:-3]
            extra_wordlist.append('have')
        if "l'"==word[:2]:
            word=word[2:]
        if '/' in word:
            tmp=word.split('/')
            if len(tmp)>2:
                extra_wordlist=extra_wordlist+tmp[1:]
            else:
                extra_wordlist.append(tmp[1])
            word=tmp[0] 
        if '-' in word:
            tmp=word.split('-')
            if len(tmp)>2:
                extra_wordlist=extra_wordlist+tmp[1:]
            else:
                extra_wordlist.append(tmp[1])
            word=tmp[0] 
        
        #punctuation clean
        word=word.translate({ord(c):None for c in removecharlist})
        
        #assign
        wordlist[wit]=word

    wordlist=[x for x in wordlist if x.isalpha()]
    extra_wordlist=[x for x in extra_wordlist if x.isalpha()]

    return wordlist+extra_wordlist

def remove_stopwords(wordlist):
    return [word for word in wordlist if word not in stop_words]

def get_wordfixlist(wordfixfile):
    wordfixes=pd.read_csv(wordfixfile,sep='\t')
    wordfixes.Fix=wordfixes.Fix.apply(lambda x:x.split() if not isinstance(x,float) else []) #converts empty fix to empty list
    return wordfixes

def reduce_to_stems(wordlist):
    porter = PorterStemmer()
    return [porter.stem(word) for word in wordlist]

def customfix(wordlist,wordfixes_df=None):
    extrawordslist=[]
    wordfixeslist=list(wordfixes_df.Word)
    for wit,word in enumerate(wordlist):
        if word in wordfixeslist:
#             print(word)
#             print(wordfixes.loc[wordfixes.Word==word,'Fix'])
            fixedwords=wordfixes_df.loc[wordfixes_df.Word==word,'Fix'].values[0]
            if len(fixedwords)==0:
                wordlist[wit]=''
            elif len(fixedwords)==1:
                wordlist[wit]=fixedwords[0]
            elif len(fixedwords)>1:
                wordlist[wit]=fixedwords[0]
                if len(fixedwords)>2:
                    extrawordslists=extrawordslist+fixedwords[1:]
                else:
                    extrawordslist.append(fixedwords[1])
    wordlist=[word for word in wordlist if word]
    if len(wordlist)==1:
        if isinstance(wordlist[0],list):
            wordlist=wordlist[0]
    return wordlist+extrawordslist

def build_word_dictionary(d_series):
    dlist=list(set(list(d_series.apply(pd.Series).stack().apply(lambda x: x[0] if isinstance(x,list) else x).values)))
    dlist.sort()
    return pd.DataFrame(dlist,columns=['word'])

def clean_corpus(data_df,column_name,wordfixfile=None):
    tmp_df=data_df[data_df[column_name]!=''].copy()
    clean_name=column_name+'clean'
    tmp_df[clean_name]=tmp_df[column_name].apply(lambda x: x.split())
    tmp_df[clean_name]=tmp_df[clean_name].apply(remove_punc)
    tmp_df[clean_name]=tmp_df[clean_name].apply(remove_stopwords)
    if wordfixfile is not None:
        wordfixes_df=get_wordfixlist(wordfixfile)
        part_customfix=partial(customfix,wordfixes_df=wordfixes_df)
        tmp_df[clean_name]=tmp_df[clean_name].apply(part_customfix)
    tmp_df[clean_name]=tmp_df[clean_name].apply(remove_stopwords)
    tmp_df=tmp_df[tmp_df[clean_name]!='']
    tmp_df[clean_name+'_stems']=tmp_df[clean_name].apply(reduce_to_stems)
    dictionary=build_word_dictionary(tmp_df[clean_name+'_stems'])
    tmp_df[clean_name+'_stems_wordvec']=tmp_df[clean_name+'_stems'].apply(lambda x: [dictionary[dictionary.word==word].index.values[0] for word in x])
    return tmp_df,dictionary
