#!/usr/bin/env python
# coding: utf-8

# ### レシーカカテゴリ自動分類用バッチ CSVファイル対応版
# - 更新日・更新者　2021年6日17日　西澤
# 
# #### フォルダ構成<br>
# - 同階層
#     - autocategory_receika_pluszero.ipynb   (本プログラム)  
#     - data
#         - 自動分類マスタ.csv
#         - 自動分類正解カテゴリデータ.csv
#     - model
#         - make_model_20210518.ipynb
#         - learning_text.txt
#         - model_daily_300dim_20210518.bin
# 
# #### 実行環境
# - 実行環境（CCCMK内）に関して
# DB: postgresql <br>
# 実行環境 OS: Ubuntu (linux) <br>
# 
# 
# - CCCMK内でのpython実行環境 <br>
#   python 3.7.6<br>
#   pandas 1.1.0<br>
#   fasttext 0.9.2<br>
#   numpy 1.18.1<br>
# 
# #### プログラム概要
# 1. 履歴データから、自動分類マスタを作成　（※POCではこちらは作成済みです。）
# 1. 自動分類マスタの商品名をベクトル化
# 1. 分類対象のリストを読み込む
# 1. 分類対象の商品名をベクトル化し、自動分類マスタのベクトルと内積
# 1. 分類対象と業態が一致し、スコアが最高のデータをカテゴリの正解データとする
# 1. 結果を出力
# 
# 
# #### 自動分類正解データ.csv のデータ加工方法<br>
# 
# receika_dd (履歴)に対して、<br>
# 1. JANコードがある商品は、JANコードに基づくカテゴリの付与 <br>
# 1. common_syohin_nm_categories <br>
# 1. common_food_syohin_nm_categories <br>
# 1. 個チェーン用のカテゴリ補正テーブル <br> 
# にて、カテゴリを付与したデータ <br>
# 
# 
# #### 自動分類の処理対象<br>
# 自動分類マスタ.csv auto_saibunrui_cd が空のレコード　<br>
# →　自動分類の処理の結果を、auto_saibunrui_cd に格納します。　<br>
#    (こちらは、新規で登録される商品名がauto_saibunrui_cd が空で登録されています。)

# In[5]:


# -*- coding: utf-8 -*-
import os
import fasttext
import numpy as np
import pandas as pd
import re 
import unicodedata
import gc
import os
import sys
import math


# #### テキスト変換の関数　（全角化）

# In[20]:


HAN_UPPER = re.compile(u"[A-Z]")
HAN_LOWER = re.compile(u"[a-z]")

def han2zen(word):
    word = HAN_UPPER.sub(lambda m: chr(ord(u"Ａ") + ord(m.group(0)) - ord("A")), word)
    word = HAN_LOWER.sub(lambda m: chr(ord(u"ａ") + ord(m.group(0)) - ord("a")), word)
    return word


# #### fasttext のモデル読み込みとワードのベクトル化・配列への格納

# In[181]:


te_df = pd.read_csv('data/自動分類正解カテゴリデータ.csv',sep = '\t')
model_path = './model/model_daily_300dim_20210518.bin'
model = fasttext.load_model(model_path)    

global ne_word_list
global c_mat
c_model = []
ne_word_list = []


for index,t in te_df.iterrows():            
    vec = model[t['syohin_nm']]
    if len(vec) != 300: print("300次元ではないベクトル")
    n2 = np.linalg.norm(vec)
    if n2 == 0 : n2 = 1
    c_model.append(vec/n2)
    ne_word_list.append([t['id'],t['syohin_nm'],t['saibunrui_cd'],t['gyotai_dai_cd'],t['tanka'],t['count']])
    
    
cc_model = np.array(c_model)
c_mat = cc_model.reshape(t['id'], 300)
del(c_model)
del(cc_model)
del(te_df)
del(vec)


# #### 自動分類対象を取得

# In[239]:


all_items_pd = pd.read_csv('data/自動分類マスタ.csv',sep = '\t')
items_pd = all_items_pd[~all_items_pd['SYOHIN_NM'].isnull()].sample(1000)

# こちらの処理は、CCCMK内では、autoed_saibunrui_cd = null のレコードを取得しています。　


# #### 類似度を使用したカテゴリの自動分類

# In[241]:


# 自動分類の処理件数
output_number = 100
write_data = []

for index,rows in items_pd.iterrows():
    target_word = rows['SYOHIN_NM']
    target_gyotai_dai_cd = rows['GYOTAI_DAI_CD']
    a_name = ""
    match_flg = 0
    
    try:
        a_name = unicodedata.normalize('NFKC', target_word ) #商品名の正規化
    except:
        continue

    a_name = han2zen(a_name)
    max_sim = 0
    
    # ターゲット商品名のベクトル化
    v1 = np.array([])
    v1 = model[a_name]
    n1 = np.linalg.norm(v1)
    c_v = np.array([])
    c_v = (v1/n1).reshape(1,300)

    # ベクトルの内積
    calc_mat = []
    calc_mat = np.dot(c_mat, c_v.transpose())
    sort_array_index = np.argsort(calc_mat,axis = 0)[::-1]
    result_index = ""

    # スコアと業態の判定
    for k in sort_array_index:
        i = k[0]
        
        if ne_word_list[i][1] is None:
            continue
            
        # 業態が A011 (外食)の場合は、カテゴリのコード体系が変わる　
        # （外食：700000 系 & 90000系、それ以外 : 大分類700000系以外を使用）ため、処理を分岐しています。
        if target_gyotai_dai_cd == 'A011':
            if str(ne_word_list[i][2])[:1] != '7' and  str(ne_word_list[i][2])[:1] != '9' and  str(ne_word_list[i][2])[:1] != '0':
                continue
        else:
            if str(ne_word_list[i][2])[:1] == '7':
                continue
        
        # 業態大分類が空の場合は、非外食とみなして、スコア最大
        if str(target_gyotai_dai_cd) == 'nan': 
            result_index = i
            break
        
        # 業態大分類の一致の判定
        if ne_word_list[i][3] == target_gyotai_dai_cd:
            result_index = i
            break
    
    score = calc_mat[result_index][0]
    n_sim = '{:.2f}'.format(score)
    s_word = ne_word_list[result_index][1]
    s_saibunrui_cd = ne_word_list[result_index][2]
    write_data.append([target_word,target_gyotai_dai_cd,s_word,n_sim,s_saibunrui_cd])

    if len(write_data) >= output_number: break


# In[237]:


write_df = pd.DataFrame(write_data, columns = ['syohin_nm','gyotai_dai_cd','hit_word','score','saibunrui_cd'])


# #### アウトプットに関して
# 
# CCCMK 社内では、アウトプットの結果をDBへ格納しています。
# （自動分類マスタのカテゴリ・スコア・ヒットしたワードを更新）
# 
# POCの際は、結果はCSVへ出力いただいて問題ございません。

# In[ ]:




