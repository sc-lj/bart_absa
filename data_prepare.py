# -*- encoding: utf-8 -*-
'''
@File    :   data_prepare.py
@Time    :   2021/12/07 18:37:26
@Author  :   lujun
@Version :   1.0
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   数据处理
'''

import pandas as pd
import json

def reshape_data(text,x):
    if "items" not in x:
        return []
    items = x['items']
    if len(items) == 0:
        return []
    text = text.strip().strip('"')
    new_items = []
    for item in items:
        line = {}
        subtext = item['abstract'].replace("<span>","").replace("</span>","").strip().strip('"')
        line['subtext'] = subtext
        subtext_index = text.find(subtext)
        
        aspect = item['prop'].strip().strip('"')
        if len(aspect):
            aspect_start = subtext.find(aspect)
            if aspect_start != -1:
                aspect_index = [subtext_index+aspect_start,subtext_index+aspect_start+len(aspect)]
            else:
                aspect_index = []
        else:
            aspect_index = []

        if len(aspect_index)==0 or len(aspect)==0 or not isinstance(aspect,str):
            continue

        line['aspect'] = aspect
        line['aspect_index'] = aspect_index

        opinion = item['adj'].strip().strip('"')
        line['opinion'] = opinion
        if len(opinion):
            opinion_start = subtext.find(opinion)
            if opinion_start != -1:
                opinion_index = [subtext_index+opinion_start,subtext_index+opinion_start+len(opinion)]
            else:
                opinion_index = []
        else:
            opinion_index = []

        if len(opinion_index)==0 or len(opinion)==0  or not isinstance(opinion,str):
            continue

        line['opinion_index'] = opinion_index
        line['sentiment'] = item['sentiment']
        new_items.append(line)
    return new_items


def get_data_index(text,items):
    if isinstance(items,str):
        items = eval(items)
    if len(items) == 0:
        return []
    text = text.strip().strip('"')
    new_items = []
    for item in items:
        line = {}
        subtext = item['subtext'].replace("<span>","").replace("</span>","").strip().strip('"')
        line['subtext'] = subtext
        subtext_index = text.find(subtext)
        
        aspect = item['aspect'].strip().strip('"')
        if len(aspect):
            aspect_start = subtext.find(aspect)
            if aspect_start != -1:
                aspect_index = [subtext_index+aspect_start,subtext_index+aspect_start+len(aspect)]
            else:
                aspect_index = []
        else:
            aspect_index = []
        if len(aspect_index)==0 or len(aspect)==0  or not isinstance(aspect,str):
            continue

        line['aspect'] = aspect
        line['aspect_index'] = aspect_index

        opinion = item['opinion'].strip().strip('"')
        line['opinion'] = opinion
        if len(opinion):
            opinion_start = subtext.find(opinion)
            if opinion_start != -1:
                opinion_index = [subtext_index+opinion_start,subtext_index+opinion_start+len(opinion)]
            else:
                opinion_index = []
        else:
            opinion_index = []
        if len(opinion_index)==0 or len(opinion)==0  or not isinstance(opinion,str):
            continue

        line['opinion_index'] = opinion_index
        line['sentiment'] = item['sentiment']
        new_items.append(line)
    return new_items

def test_1():
    # 数据1
    # 情感0为负，1为中，2为正
    print("处理数据1")
    c3_1 = pd.read_csv("./data/noise_data/3C_clean.csv")
    c3_1 = c3_1['0'].map(eval)
    c3_1 = pd.DataFrame(c3_1.values.tolist())
    c3_1['topic_list'] = c3_1[["text",'result']].apply(lambda x:reshape_data(x[0],x[1]),axis=1)
    c3_1.drop(columns=['result'],inplace=True)

    c3_2 = pd.read_csv("./data/noise_data/3C百度API结果清洗0616.csv")
    c3_2['new_topic_list'] = c3_2[['text','topic_list']].apply(lambda x:get_data_index(x[0],x[1]),axis=1)
    c3_2.drop(columns=['topic_list'],inplace=True)
    c3_2.rename(columns={"new_topic_list":"topic_list"},inplace=True)
    c3_2.dropna(inplace=True)
    c3 = pd.concat([c3_2,c3_1]).drop_duplicates(subset=['text'])
    c3['text'] = c3['text'].apply(lambda x:x.strip().strip('"'))
    print(c3.shape)
    return c3

# 数据2
def test_2():
    print("处理数据2")
    # baidu_api_1 = pd.read_excel("./data/noise_data/百度API结果清洗.xlsx")
    # baidu_api_1.drop(columns=["is_empty"],inplace=True)
    # baidu_api_1['topic_list'] = baidu_api_1[['text','topic_list']].apply(lambda x:get_no_subtext_data_index(x[0],x[1]),axis=1)

    baidu_api_2 = pd.read_csv("./data/noise_data/百度API结果清洗0616.csv")
    baidu_api_2.drop(columns=["is_empty"],inplace=True)
    baidu_api_2['new_topic_list'] = baidu_api_2[['text','topic_list']].apply(lambda x:get_data_index(x[0],x[1]),axis=1)
    baidu_api_2.drop(columns=['topic_list'],inplace=True)
    baidu_api_2.rename(columns={"new_topic_list":"topic_list"},inplace=True)
    baidu_api_2.dropna(inplace=True)

    baidu_api_3 = []
    import json
    with open("./data/noise_data/comment_tag1.txt",'r') as f:
        for line in f.readlines():
            baidu_api_3.append(json.loads(line.strip()))
    baidu_api_3 = pd.DataFrame(baidu_api_3)
    baidu_api_3 = baidu_api_3.dropna()
    baidu_api_3['topic_list'] = baidu_api_3[["text",'result']].apply(lambda x:reshape_data(x[0],x[1]),axis=1)
    baidu_api_3.drop(columns=['result'],inplace=True)

    baidu_api_4 = pd.read_csv("./data/noise_data/hotel_clean.csv")
    baidu_api_4 = baidu_api_4['0'].map(eval)
    baidu_api_4= pd.DataFrame(baidu_api_4.values.tolist())
    baidu_api_4['topic_list'] = baidu_api_4[["text",'result']].apply(lambda x:reshape_data(x[0],x[1]),axis=1)
    baidu_api_4.drop(columns=['result'],inplace=True)

    baidu_api = pd.concat([baidu_api_2,baidu_api_3,baidu_api_4]).drop_duplicates(subset=['text'])
    baidu_api['text'] = baidu_api['text'].apply(lambda x:x.strip().strip('"'))
    print(baidu_api.shape)
    return baidu_api

# 数据3
def test_3():
    print("处理数据3")
    canying_baidu_api_1 = pd.read_csv("./data/noise_data/餐饮百度API结果清洗0616.csv")
    canying_baidu_api_1['new_topic_list'] = canying_baidu_api_1[['text','topic_list']].apply(lambda x:get_data_index(x[0],x[1]),axis=1)
    canying_baidu_api_1.drop(columns=['topic_list'],inplace=True)
    canying_baidu_api_1.rename(columns={"new_topic_list":"topic_list"},inplace=True)
    canying_baidu_api_1.dropna(inplace=True)

    canying_baidu_api_2 = []
    import json
    with open("./data/noise_data/comment_restruant.txt",'r') as f:
        for line in f.readlines():
            canying_baidu_api_2.append(json.loads(line.strip()))
    canying_baidu_api_2 = pd.DataFrame(canying_baidu_api_2)
    canying_baidu_api_2 = canying_baidu_api_2.dropna()
    canying_baidu_api_2['topic_list'] = canying_baidu_api_2[['text','result']].apply(lambda x:reshape_data(x[0],x[1]),axis=1)
    canying_baidu_api_2.drop(columns=['result'],inplace=True)

    canying_baidu_api_3 = pd.read_csv("./data/noise_data/restruant_clean.csv")
    canying_baidu_api_3 = canying_baidu_api_3['0'].map(eval)
    canying_baidu_api_3= pd.DataFrame(canying_baidu_api_3.values.tolist())
    canying_baidu_api_3['topic_list'] = canying_baidu_api_3[["text",'result']].apply(lambda x:reshape_data(x[0],x[1]),axis=1)
    canying_baidu_api_3.drop(columns=['result'],inplace=True)

    canying_baidu_api = pd.concat([canying_baidu_api_1,canying_baidu_api_2,canying_baidu_api_3]).drop_duplicates(subset=['text'])
    canying_baidu_api['text'] = canying_baidu_api['text'].apply(lambda x:x.strip().strip('"'))
    print(canying_baidu_api.shape)
    return canying_baidu_api

def get_hotel_topic(hotel_):
    groups = hotel_.groupby("text")
    result = []
    for text,group in groups:
        new_terms = []
        for xx in group.values.tolist():
            line = {}
            line['subtext'] = xx[0]
            line['aspect'] = xx[1]
            line['aspect_index'] = [xx[2],xx[3]]
            line['opinion'] = xx[4]
            line['opinion_index'] = [xx[5],xx[6]]
            line['sentiment'] = xx[7]
            new_terms.append(line)
        result.append({"text":text,"topic_list":new_terms})
    return result 

# 数据4
def test_4():
    print("处理数据4")
    hotel_ = pd.read_excel("./data/noise_data/合并_有效_19925条&无效_75条_10.09修改 (1).xlsx")
    hotel_ = hotel_.dropna()
    hotel = get_hotel_topic(hotel_)
    hotel = pd.DataFrame(hotel)
    print(hotel.shape)
    return hotel

# 数据5
def test_5():
    print("处理数据5")
    phone = []
    import json
    with open("./data/noise_data/phone_comment_tag.txt",'r') as f:
        for line in f.readlines():
            phone.append(json.loads(line.strip()))
    phone = pd.DataFrame(phone)
    phone = phone.dropna()
    phone['topic_list'] = phone[['text','result']].apply(lambda x:reshape_data(x[0],x[1]),axis=1)
    phone.drop(columns=['result'],inplace=True)
    phone['text'] = phone['text'].apply(lambda x:x.strip().strip('"'))
    print(phone.shape)
    return phone

def find_subtext(x):
    text,triplets = x
    terms = []
    # print(text)
    for triplet in eval(triplets):
        line = {}
        # print(triplet)
        aspect,opinion,sentiment = triplet
        if len(aspect)==0:
            continue
        else:
            line['aspect'] = text[int(aspect[0]):int(aspect[1])]
            line['aspect_index'] = [int(aspect[0]),int(aspect[1])]
        if len(opinion) == 0:
            continue
        else:
            line['opinion'] = text[int(opinion[0]):int(opinion[1])]
            line['opinion_index'] = [int(opinion[0]),int(opinion[1])]
        line['sentiment'] = sentiment[0]
        line['subtext'] = text
        terms.append(line)
    return terms


# 数据6
def test_6():
    print("处理数据6")
    checkpoint_1 = pd.read_csv("./data/noise_data/test-checkpoint.csv")
    checkpoint_1 = checkpoint_1.dropna()
    checkpoint_1['topic_list']=checkpoint_1[['text','triplet']].apply(lambda x:find_subtext(x),axis=1)
    checkpoint_1.drop(columns=['triplet','length'],inplace=True)

    checkpoint_2 = pd.read_csv("./data/noise_data/train.csv")
    checkpoint_2 = checkpoint_2.dropna()
    checkpoint_2['topic_list']=checkpoint_2[['text','triplet']].apply(lambda x:find_subtext(x),axis=1)
    checkpoint_2.drop(columns=['triplet','length'],inplace=True)

    checkpoint = pd.concat([checkpoint_1,checkpoint_2])
    print(checkpoint.shape)
    return checkpoint


c3 = test_1()
baidu_api = test_2()
canying_baidu_api = test_3()
hotel = test_4()
phone = test_5()
checkpoint = test_6()


print("合并数据")
all_data = pd.concat([c3,baidu_api,canying_baidu_api,hotel,phone,checkpoint])
all_data = all_data.drop_duplicates(subset=['text'])
all_data = all_data.dropna()
print(all_data.shape)
all_data = all_data[all_data['topic_list'].apply(lambda x:len(x)>0)]
print(all_data.shape)

chinses_punctuation = '?？；：:;。！!,，'
def find_text_split(text,index):
    """[根据索引查找前后标点符号]

    Args:
        text ([type]): [description]
        index ([type]): [description]
    """
    punc_index = index
    for i in range(index,len(text)):
        if text[i] in chinses_punctuation:
            punc_index = i
            break
    return punc_index+1
    

def change_text_length(text,result):
    new_line = []
    root_index = 0
    threshold = 400
    result = sorted(result,key=lambda x:min(x['aspect_index'][-1],x['opinion_index'][-1]))
    while root_index<len(text):
        while True:
            # 第一次切断
            try:
                backup = [ x for x in result if max(x['aspect_index'][-1],x['opinion_index'][-1])<threshold+root_index][-1]
                break
            except:
                for i in range(root_index,len(text)):
                    if text[i] in chinses_punctuation:
                        root_index = i
                        break
                root_index += 1
                continue
        backup_index = max(backup['aspect_index'][-1],backup['opinion_index'][-1])
        # 找到第一次切断后，标点符号的索引
        new_index = old_index = find_text_split(text,int(backup_index))
        # 根据标点符号索引切断
        first = [ x for x in result if max(x['aspect_index'][-1],x['opinion_index'][-1])<new_index]
        other = [ x for x in result if min(x['aspect_index'][-1],x['opinion_index'][-1])>=new_index]

        if len(other)==0:
            new_index=len(text)
        new_line.append({"text":text[root_index:new_index],"topic_list":first})
        result_ = []
        # 对索引进行纠偏
        for l in first:
            aspect_index = l['aspect_index']
            aspect_index = [aspect_index[0]-root_index,aspect_index[1]-root_index]
            l['aspect_index'] = aspect_index

            opinion_index = l['opinion_index']
            opinion_index = [opinion_index[0]-root_index,opinion_index[1]-root_index]
            l['opinion_index'] = opinion_index
            result_.append(l)
        result = other
        root_index = new_index
        if len(result):
            last_index = min(result[0]['aspect_index'][-1],result[0]['opinion_index'][-1])
            if last_index - root_index> threshold:
                threshold = last_index - root_index+10


    return new_line

new_data = []
for line in all_data.to_dict(orient='records'):
    text = line['text']
    result = line['topic_list']
    if len(text)>512:
        new_line = change_text_length(text,result)
        new_data.extend(new_line)
        pass
    else:
        new_data.append(line)

all_data = pd.DataFrame(new_data)

all_data.to_csv("./data/all_data.csv",index=False,encoding="utf-8")

train_data = all_data.sample(frac=0.9)
val_data = all_data[~all_data.index.isin(train_data.index)]

train_data.to_csv("./data/vocust/train_data.csv",index=False,encoding="utf-8")
val_data.to_csv("./data/vocust/dev_data.csv",index=False,encoding="utf-8")

