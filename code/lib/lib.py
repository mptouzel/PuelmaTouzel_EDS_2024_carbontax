import numpy as np
import pandas as pd
from functools import partial
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from googletrans import Translator, constants  # pip install googletrans==4.0.0-rc1
import time
import random


def label_response(row, resp_dict):
    for key, val in resp_dict.items():
        if row[key]:
            row['res'] = row[key]
            row['restype'] = val
    return row


def get_resdf(df, resp_dict):
    resdf = df.loc[:, ['responseid'] + list(resp_dict.keys())]
    resdf['res'] = ''
    resdf['restype'] = ''
    part_lr = partial(label_response, resp_dict=resp_dict)
    resdf = resdf.apply(part_lr, axis=1)
    resdf = resdf.drop(columns=list(resp_dict.keys()))
    resdf = resdf.rename(columns={'responseid': 'id'})
    print(resdf.restype.value_counts())
    return resdf


def convert_partyvote_to_partisanvote(party):
    if party == 'Conservative Party' or party == "People's Party":
        partisan = 'conservative'
    elif party == 'Liberal Party' or party == 'ndp' or party == 'Green Party' or party == 'Bloc Québécois':
        partisan = 'progressive'
    elif party == 'I would not vote':
        partisan = 'no vote'
    else:
        print('not a standard response!')
    return partisan


def convert_year_to_decade(age):
    if age == '100 or older':
        return '60+'
    else:
        age = int(age)
        if age < 19:
            return '<19'
        elif age < 30:
            return '19-29'
        elif age < 40:
            return '30-39'
        elif age < 50:
            return '40-49'
        elif age < 60:
            return '50-59'
        elif age >= 60:
            return '60+'
        else:
            print(age)


def convert_livingenv(livenv):
    if livenv == 'Within a large city':
        return 'urban'
    elif livenv == 'Within a suburb, adjacent to a large city' or livenv == 'In a smaller, regional city':
        return 'suburban'
    elif livenv == 'In a small town':
        return 'smalltown'
    elif livenv == 'In a rural area' or livenv == 'In a remote area':
        return 'ruralremote'
    elif livenv == 'Not sure':
        return 'notsure'
    else:
        print(livenv)


def convert_partisanship(specscore):
    if specscore == 'Centre5':
        return 'none'
    elif specscore == 'Far to the right10':
        return 'right'
    elif specscore == 'Far to the left0':
        return 'left'
    elif specscore == 'Not sure':
        return 'notsure'
    else:
        if int(specscore) < 5:
            return 'left'
        else:
            return 'right'


def convert_caruse(modetoworkschool):
    if modetoworkschool == 'Drive alone':
        return 'driver'
    elif modetoworkschool == 'Drive with others or carpool':
        return 'commuter'
    elif modetoworkschool == 'Transit' or modetoworkschool == 'Walk' or \
            modetoworkschool == 'Cycle' or modetoworkschool == 'Work/study at home' or \
            modetoworkschool == "This doesn't apply to me":
        return 'none'


def convert_type(df):
    df['agedec'] = df['ageyear'].apply(convert_year_to_decade)
    df = df.drop(columns='ageyear')
    df['livingenv'] = df['livingenv'].apply(convert_livingenv)
    df['partisanship'] = df['partisanship'].apply(convert_partisanship)
    df['caruse'] = df['caruse'].apply(convert_caruse)
    df['partyvote'] = df['partyvote'].apply(convert_partyvote_to_partisanvote)
    return df


removecharlist = string.punctuation + (
    "".join(np.arange(10).astype(str))
)  # "!?#.,\'()`%+&:-;"+'"'

stop_words = stopwords.words("english")
stop_words = [x.translate({ord(c): None for c in removecharlist}) for x in stop_words]


def remove_punc(wordlist):

    #     print(wordlist)
    extra_wordlist = []
    for wit, word in enumerate(wordlist):
        word = word.lower()
        word = word.replace("′", "'")
        word = word.replace("`", "'")
        word = word.replace("’", "'")
        if set(word).pop() == "$":
            word = "money"
        if word[:-2] == "'s":
            word = word[:-2]
        if "n't" == word[:-3]:
            word = word[:-3]
            extra_wordlist.append("not")
        if "'ve" == word[:-3]:
            wordlist[wit] = word[:-3]
            extra_wordlist.append("have")
        if "l'" == word[:2]:
            word = word[2:]
        if "/" in word:
            tmp = word.split("/")
            if len(tmp) > 2:
                extra_wordlist = extra_wordlist + tmp[1:]
            else:
                extra_wordlist.append(tmp[1])
            word = tmp[0]
        if "-" in word:
            tmp = word.split("-")
            if len(tmp) > 2:
                extra_wordlist = extra_wordlist + tmp[1:]
            else:
                extra_wordlist.append(tmp[1])
            word = tmp[0]

        # punctuation clean
        word = word.translate({ord(c): None for c in removecharlist})

        # assign
        wordlist[wit] = word

    wordlist = [x for x in wordlist if x.isalpha()]
    extra_wordlist = [x for x in extra_wordlist if x.isalpha()]

    return wordlist + extra_wordlist


def remove_stopwords(wordlist):
    return [word for word in wordlist if word not in stop_words]


def get_wordfixlist(wordfixfile):
    wordfixes = pd.read_csv(wordfixfile, sep="\t")
    wordfixes.Fix = wordfixes.Fix.apply(
        lambda x: x.split() if not isinstance(x, float) else []
    )  # converts empty fix to empty list
    return wordfixes


def reduce_to_stems(wordlist):
    porter = PorterStemmer()
    return [porter.stem(word) for word in wordlist]


def customfix(wordlist, wordfixes_df=None):
    extrawordslist = []
    wordfixeslist = list(wordfixes_df.Word)
    for wit, word in enumerate(wordlist):
        if word in wordfixeslist:
            fixedwords = wordfixes_df.loc[wordfixes_df.Word == word, "Fix"].values[0]
            if len(fixedwords) == 0:
                wordlist[wit] = ""
            elif len(fixedwords) == 1:
                wordlist[wit] = fixedwords[0]
            elif len(fixedwords) > 1:
                wordlist[wit] = fixedwords[0]
                if len(fixedwords) > 2:
                    extrawordslist = extrawordslist + fixedwords[1:]
                else:
                    extrawordslist.append(fixedwords[1])
    wordlist = [word for word in wordlist if word]
    if len(wordlist) == 1:
        if isinstance(wordlist[0], list):
            wordlist = wordlist[0]
    return wordlist + extrawordslist


def build_vocabulary(d_series):
    strSer = partial(pd.Series, dtype=str)
    dlist = list(
        set(
            list(
                d_series.apply(strSer)
                .stack()
                .apply(lambda x: x[0] if isinstance(x, list) else x)
                .values
            )
        )
    )
    dlist.sort()
    return pd.DataFrame(dlist, columns=["word"])


def translate_responses(french_responses):
    translated_responses = french_responses.copy()
    translator = Translator()
    st = time.time()
    # iter = -1
    for ind, french_response in french_responses.iteritems():
        # iter += 1
        # print(iter)
        # if iter % int(len(nonempty_french_responses) / 10) == 0:
        #     print(str(round(iter / len(french_responses) * 100)) + "% ", end="")
        done = False
        while not done:
            try:
                translated_responses[ind] = translator.translate(
                    french_response, src="fr", dest="en"
                ).text
                done = True
            except AttributeError:
                print(str(ind) + ' failed. Resetting translator and trying again')
                time.sleep(1)
                translator = Translator()
    print("finished in " + str(time.time() - st))
    return translated_responses

# def translate_responses(df, res_name):
#     translator = Translator()
#     french_responses = df.loc[df.lang == "FR", res_name]
#     nonempty_french_responses = french_responses[french_responses != ""]
#     print(
#         "translating " +
#         str(len(nonempty_french_responses)) +
#         " non-empty responses of " +
#         str(len(french_responses)) +
#         " french responses"
#     )
#     st = time.time()
#     iter = -1
#     for ind, french_response in nonempty_french_responses.iteritems():
#         iter += 1
#         # print(iter)
#         # if iter % int(len(nonempty_french_responses) / 10) == 0:
#         #     print(str(round(iter / len(french_responses) * 100)) + "% ", end="")
#         df.loc[ind, res_name] = translator.translate(
#             french_response, src="fr", dest="en"
#         ).text
#     print("finished in " + str(time.time() - st))
#     return df


def clean_corpus(responses, wordfixfile=None, reduce_to_stems_flag=True):
    assert np.all(responses != ""), "some responses are empty!"
    cleaned_responses = responses.copy()
    cleaned_responses = cleaned_responses.apply(lambda x: x.split()).apply(remove_punc).apply(remove_stopwords)
    if wordfixfile is not None:
        wordfixes_df = get_wordfixlist(wordfixfile)
        part_customfix = partial(customfix, wordfixes_df=wordfixes_df)
        cleaned_responses = cleaned_responses.apply(part_customfix)
    cleaned_responses = cleaned_responses.apply(remove_stopwords)
    if reduce_to_stems_flag:
        cleaned_responses = cleaned_responses.apply(reduce_to_stems)
    vocabulary = build_vocabulary(cleaned_responses)
    cleaned_responses = cleaned_responses.apply(lambda x: ' '.join(x))
    return cleaned_responses, vocabulary


# def clean_corpus(tmp_df, column_name, wordfixfile=None, occurs_in_morethan=0, reduce_to_stems_flag=True):
#     # tmp_df = data_df[data_df[column_name] != ""].copy()
#     clean_name = column_name + "clean"
#     tmp_df[clean_name] = tmp_df[column_name].apply(lambda x: x.split())
#     tmp_df[clean_name] = tmp_df[clean_name].apply(remove_punc)
#     tmp_df[clean_name] = tmp_df[clean_name].apply(remove_stopwords)
#     if wordfixfile is not None:
#         wordfixes_df = get_wordfixlist(wordfixfile)
#         part_customfix = partial(customfix, wordfixes_df=wordfixes_df)
#         tmp_df[clean_name] = tmp_df[clean_name].apply(part_customfix)
#     tmp_df[clean_name] = tmp_df[clean_name].apply(remove_stopwords)
#     tmp_df = tmp_df[tmp_df[clean_name] != ""]
#     stemstr=""
#     if reduce_to_stems_flag:
#         stemstr="_stems"
#         tmp_df[clean_name + stemstr] = tmp_df[clean_name].apply(reduce_to_stems)
#     dictionary = build_word_dictionary(tmp_df[clean_name + stemstr])
#     tmp_df[clean_name + stemstr+"_wordvec"] = tmp_df[clean_name + stemstr].apply(
#         lambda x: [dictionary[dictionary.word == word].index.values[0] for word in x]
#     )
#     strSer = partial(pd.Series, dtype=str)
#     if occurs_in_morethan > 0:
#         document_occurence = (
#             tmp_df[clean_name + stemstr+"_wordvec"]
#             .apply(lambda x: list(set(x)))
#             .apply(strSer)
#             .unstack()
#             .value_counts()
#             .sort_values(ascending=False)
#             .reset_index()
#         )
#         document_occurence = document_occurence.rename(
#             columns={0: "doc_count", "index": "wordindex"}
#         )
#         document_occurence.wordindex = document_occurence.wordindex.astype(int)
#         # apply selection
#         keepword_inds = document_occurence.loc[
#             document_occurence.doc_count > occurs_in_morethan, "wordindex"
#         ].values
#         # relabel and reindex
#         tmp_df[clean_name + stemstr+"_wordvec_keep"] = tmp_df[
#             clean_name + stemstr+"_wordvec"
#         ].apply(lambda x: [wordind for wordind in x if wordind in keepword_inds])
#         tmp_df[clean_name + stemstr+"_keep"] = tmp_df[
#             clean_name + stemstr+"_wordvec_keep"
#         ].apply(
#             lambda x: [
#                 dictionary.loc[wordind][0] for wordind in x
#             ]
#         )

#         dictionary = build_word_dictionary(tmp_df[clean_name + stemstr+"_keep"])

#         tmp_df[clean_name + stemstr+"_keepwordvec"] = tmp_df[clean_name + stemstr+"_keep"].apply(
#             lambda x: [
#                 dictionary[dictionary.word == word].index.values[0] for word in x
#             ]
#         )
#     tmp_df.drop(columns=[clean_name + stemstr+"_wordvec_keep"])
#     tmp_df = tmp_df[tmp_df[column_name] != ""].copy()
#     return tmp_df, dictionary


def justclean_corpus(data_df, column_name, wordfixfile=None, occurs_in_morethan=0):
    tmp_df = data_df[data_df[column_name] != ""].copy()
    clean_name = column_name + "clean"
    tmp_df[clean_name] = tmp_df[column_name].apply(lambda x: x.split())
    tmp_df[clean_name] = tmp_df[clean_name].apply(remove_punc)
    tmp_df[clean_name] = tmp_df[clean_name].apply(remove_stopwords)
    if wordfixfile is not None:
        wordfixes_df = get_wordfixlist(wordfixfile)
        part_customfix = partial(customfix, wordfixes_df=wordfixes_df)
        tmp_df[clean_name] = tmp_df[clean_name].apply(part_customfix)
    tmp_df[clean_name] = tmp_df[clean_name].apply(remove_stopwords)
    tmp_df = tmp_df[tmp_df[clean_name] != ""]
    tmp_df[clean_name + "_stems"] = tmp_df[clean_name].apply(reduce_to_stems)
    tmp_df[clean_name + "_stems"] = tmp_df[clean_name + "_stems"].apply(lambda x: ' '.join(x))
    tmp_df = tmp_df.reset_index(drop=True)
    return tmp_df


def pairscramble_corpus(series, number_of_swaps, seed=0):
    random.seed(seed)
    num_words = 0
    row_vec = range(len(series))
    for row_id in row_vec:
        num_words += len(series[row_id])
    swap_number = 0
    tmpdat = series.apply(lambda x: x.split(' ')).copy()
    while swap_number < number_of_swaps:
        draw_rowpair = random.sample(row_vec, 2)
        draw_wordindpair = [random.sample(range(len(tmpdat.iloc[row_id])), 1)[0] for row_id in draw_rowpair]
        wordpair = [tmpdat[row_id][wordind] for row_id, wordind in zip(draw_rowpair, draw_wordindpair)]
    #     countpair=[tmpdat.iloc[row_id][1][wordind] for row_id,wordind in zip(draw_rowpair,draw_wordindpair)]
        tmpdat.iloc[draw_rowpair[0]][draw_wordindpair[0]] = wordpair[1]
        tmpdat.iloc[draw_rowpair[1]][draw_wordindpair[1]] = wordpair[0]
    #     tmpdat.iloc[draw_rowpair[0]][1][draw_wordindpair[0]]=countpair[1]
    #     tmpdat.iloc[draw_rowpair[1]][1][draw_wordindpair[1]]=countpair[0]
        swap_number += 1
    tmpdat = tmpdat.apply(lambda x: ' '.join(x))
    return tmpdat
