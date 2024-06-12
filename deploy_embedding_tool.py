# -*- coding: utf-8 -*-
from langchain.embeddings import HuggingFaceEmbeddings
import sentence_transformers
import sys
import os
from loguru import logger
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import jieba  
import math


import datetime
from loguru import logger

current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_format = f"<green>{current_date}</green> <level>{{message}}</level>"

logger.add("deploy_embedding_tool.log", format=log_format, level="INFO", rotation="50 MB")

  
# 停用词列表  
all_stopwords = set(["我要", "我", "您","麻烦", "介绍","可不可以","不可以","可以","能","不能","想","人","哪些","那些",
                     "的", "了", "在", "是", "有", "都", "就", "不", "一个", "也", "找",
                     "要", "去”, “说", "很", "都", "会", "看", "着", "吧", "等", "没", "到", "做", "再", "可以", "如果",
                       "自己", "你", "最", "这个", "那么", "又", "就是", "只", "但是", "因为", "它", "一些","一下", "还有", "觉得", 
                       "出来", "现在", "开始", "确实", "什么", "很多", "还是", "已经", "自己","?","？","。",".",",","，"])  
 
# jieba.load_userdict("VECTOR_RECALL/data/my_dict.txt") 
jieba.add_word("任我选")        
"""
import nltk  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  

# 设置NLTK数据目录的环境变量  
nltk_data_path = '/data/staryea/aigc/aigc-server-chat/VECTOR_RECALL'  
os.environ['NLTK_DATA'] = nltk_data_path  
# 确保已经下载了punkt分词器和停用词列表  
nltk.download('punkt')  
nltk.download('stopwords')

# 自定义的停用词列表  
custom_stopwords = set(["我要", "我", "麻烦", "介绍"])  
  
# 合并NLTK的停用词列表和自定义停用词列表  
all_stopwords = set(stopwords.words('chinese')) | custom_stopwords  
"""
def get_packages(file_path):
    # 读取Excel文件的三个工作表
    sheets = pd.read_excel(file_path, sheet_name=["Q分配一组", "Q分配二组", "QA"])

    # 读取Q列的数据并生成套餐列表
    package_list = []
    for sheet_name, df in sheets.items():
        print("sheet_name:", sheet_name)
        if "Q" in df.columns:
            q_column = df["Q"].tolist()
            package_list.extend(q_column)
    return package_list

# 去除停用词的函数  
def remove_stopwords(text, stopwords):  
    # words = word_tokenize(text)  # nltk分词  
    # 添加多个自定义词
    # custom_words = ["任我选"]
    # for word in custom_words:
    #     jieba.add_word(word)

     # 使用jieba进行分词  
    words = jieba.cut(text, cut_all=False)  
    filtered_words = [word for word in words if word not in stopwords]  # 去除停用词  
    return ''.join(filtered_words)  # 重新组合成字符串  
  

def similarity_penalty(s1_length, s2_length):
    """
    计算句子s1和s2的长度差距的惩罚因子。
    
    参数:
    s1_length (int): 句子s1的长度。
    s2_length (int): 句子s2的长度。
    
    返回:
    float: 在0到1之间的惩罚因子。
    """
    # 定义惩罚因子的上限
    MAX_PENALTY = 0.5
    
    # 计算两个句子长度的绝对差值
    length_diff = abs(s1_length - s2_length)
    
    # 定义两个不同的衰减率
    k_slow = 0.005  # 慢衰减率，用于长度差小于等于10
    k_fast = 2   # 快衰减率，用于长度差大于10
    
    # 根据长度差选择衰减率
    if length_diff <= 20:
        # 当长度差小于或等于10时，使用慢衰减率
        penalty_factor = 1 - math.exp(-k_slow * length_diff)
    else:
        # 当长度差大于10时，使用快衰减率
        penalty_factor = 1 - math.exp(-k_fast * (length_diff - 20))
    
    # 确保惩罚因子不超过MAX_PENALTY
    penalty_factor = min(penalty_factor, MAX_PENALTY)
    
    # 确保惩罚因子在0到1之间
    return 1-max(0, penalty_factor)

def search_similar_packages(query, packages, limit=8):
    """
    搜索与给定查询最相似的套餐名称
    """
    results = process.extract(query, packages, limit=limit)
    return [result for result in results]

def search_similar_packages_with_score(Q, result, limit=5):
    # txt = remove_stopwords(Q, all_stopwords)
    txt = Q
    # 打印结果
    # print(f"\n Query: {Q}")
    logger.info(f"txt: {txt}")

    #返回结果
    return_result = []

    id_score_mapping = {}  # 存储 'id' 和 'score' 的对应关系
    id_content_mapping = {}  # 存储 'id' 和 'score' 的对应关系
    score_list = []#余弦相似度
    txt_packages = []
    for hits_new in result:
        if 'question' not in hits_new:
            continue
        doc_id = hits_new['id']
        content = hits_new['question']
        score = hits_new['score']
        id_score_mapping[doc_id] = score
        id_content_mapping[doc_id] = content
        score_list.append(score)
        txt_packages.append(content)
    
    if len(txt_packages) == 0 or len(score_list) == 0:
        return return_result
    
    content_id_mapping = {v: k for k, v in id_content_mapping.items()}  # 反向存储 'id' 和 'content' 
    # 转换为 NumPy 数组以进行筛选和归一化
    score_array = np.array(score_list)
    # print("score_array:",score_array[:10])
    s_score_array = score_array[:10]
    logger.info(f"score_array: {s_score_array}")

        # 相似度筛选
    similar_scores_filtered = score_array[score_array > 15]    
    # print("similar_scores_filtered:", similar_scores_filtered[:10])
    s_similar_scores_filtered = similar_scores_filtered[:10]
    logger.info(f"similar_scores_filtered:{s_similar_scores_filtered}")


    merge_text_scores = {}
    merge_id_scores = {}
    if len(similar_scores_filtered) > 0:                
        # 归一化筛选后的分值
        min_score = np.min(similar_scores_filtered)
        max_score_cos = np.max(similar_scores_filtered)
        normalized_scores = (similar_scores_filtered - min_score) / (max_score_cos - min_score)

        if min_score != max_score_cos:
            normalized_scores = (similar_scores_filtered - min_score) / (max_score_cos - min_score)
        else:
            # 处理最大值和最小值相等的情况
            normalized_scores = np.ones_like(similar_scores_filtered)

        
        # 获取排序前五的归一化分值和对应的 id
        sorted_indices = np.argsort(normalized_scores)[::-1][:limit]
        sorted_scores = normalized_scores[sorted_indices]
        sorted_ids = np.array(list(id_score_mapping.keys()))[sorted_indices]
        # print("sorted_ids: ",sorted_ids)
        logger.info(f"sorted_ids: {sorted_ids}")
        # 输出排序前五的 id 和归一化分值
        for doc_id, score in zip(sorted_ids, sorted_scores):
            # print(f"ID: {doc_id}, Normalized Score: {score}")
            logger.info(f"ID: {doc_id}, Normalized Score: {score}")
           
            content = id_content_mapping.get(doc_id)
            
            fuzz_ratio = fuzz.partial_ratio(txt, content)
            logger.info(f"txt: {txt}, content: {content}, fuzz_ratio: {fuzz_ratio}")
            
            if fuzz_ratio > 50:
                merge_id_scores[doc_id] = merge_id_scores.get(doc_id, 0) + score
                merge_text_scores[content] = merge_text_scores.get(content, 0) + score

    else:
        # print(f"\n Query: {query}")
        # print(f"txt: {txt}")
        # print("cos Most:")
        # print("No similar packages found.")
        logger.info("cos Most:")
        logger.info("No similar packages found.")
        return []
    

    similar_packages = search_similar_packages(txt, txt_packages, limit)  
    # print("similar_packages:",similar_packages) 
    # 获取最大值和最小值
    min_score = min(sim_score for _, sim_score in similar_packages)
    max_score_fuzz = max(sim_score for _, sim_score in similar_packages)

    # 归一化筛选后的分值
    if max_score_fuzz != min_score:
        normalized_packages = [(sim_txt, similarity_penalty(len(sim_txt), len(txt))*((sim_score - min_score) / (max_score_fuzz - min_score))) for sim_txt, sim_score in similar_packages]
    else:
        # 处理最大值和最小值相等的情况
        normalized_packages = [(sim_txt, similarity_penalty(len(sim_txt), len(txt))*1.0) for sim_txt, _ in similar_packages]

    # 对归一化后的分值进行排序
    sorted_packages = sorted(normalized_packages, key=lambda x: x[1], reverse=True)
    # print("sorted_packages:", sorted_packages)
    logger.info(f"sorted_packages:{sorted_packages}")
    # 打印结果
    # print(f"Query: {query}")
    # print(f"txt: {txt}")
    # print("fuzz Most:")
    for index, (sim_txt, sim_score) in enumerate(sorted_packages):
        if sim_score>0.5:
            merge_text_scores[sim_txt] = merge_text_scores.get(sim_txt, 0) + sim_score   
            # print(f"{index+1}. {sim_txt} (Normalized Score: {sim_score})")
            logger.info(f"{index+1}. {sim_txt} (Normalized Score: {sim_score})")

    if not merge_text_scores:
        # print("No similar packages found.")
        # print("==============================")
        logger.info("No merge packages found.")
        logger.info("==============================")
        return return_result
    
    #筛选一条数据的条件判断
    max_score_similar_packages = max(similar_packages, key=lambda x: x[1])
    max_content_fuzz = max_score_similar_packages[0]
    max_score_fuzz = max_score_similar_packages[1]

    max_score_index = np.argmax(score_array)
    max_score_cos = score_array[max_score_index]
    max_content_cos = txt_packages[max_score_index]
    # print("max_score_cos:", max_score_cos,"max_content_cos:", max_content_cos)
    # print("max_score_fuzz:", max_score_fuzz,"max_content_fuzz:", max_content_fuzz)
    logger.info(f"max_score_cos:{max_score_cos},max_content_cos:{max_content_cos}")
    logger.info(f"max_score_fuzz:{max_score_fuzz},max_content_fuzz:{max_content_fuzz}")
    highest_flag = False
    if max_content_cos==max_content_fuzz and max_score_cos==max_score_fuzz:
        highest_flag = True
    # print("highest_flag:", highest_flag)
    logger.info(f"highest_flag: {highest_flag} " )

    merged_scores = {}
    for text, score in merge_text_scores.items():
        merged_scores[text] = merged_scores.get(text, 0) + score
    if not merged_scores:
        # print("No similar packages found.")
        # print("==============================")
        logger.info("No similar packages found.")
        logger.info("==============================")
        return return_result
    else:
        sorted_scores = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
        top_results = sorted_scores[:5]
        # print("====Top Results:")
        logger.info("====Top Results:")
        for i, (text, score) in enumerate(top_results):
            # print(f"{i+1}: {text} (Total Score: {score})")
            logger.info(f"{i+1}: {text} (Total Score: {score})")

        # 提取分值并归一化
        max_score = max(score for _, score in top_results)
        min_score = min(score for _, score in top_results)
        if max_score != min_score:
            normalized_scores = [(score - min_score) / (max_score - min_score) for _, score in top_results]
        else:
            # 处理最大值和最小值相等的情况
            normalized_scores = [1.0 for _, _ in top_results]

        
        # print("normalized_scores:", normalized_scores)
        logger.info(f"normalized_scores: {normalized_scores}")
        
        # 计算熵值
        epsilon = 1e-10  # 避免出现0或负数的情况
        entropy = -sum(score * math.log(score + epsilon) for score in normalized_scores)

        # 打印熵值
        # print(f"Entropy: {entropy}")
        logger.info(f"Entropy: {entropy}")
        
        if entropy <= 0 or min_score > 0.9:
            # 输出归一化后的结果
            # print("Normalized Results:")
            logger.info("Normalized Results:")
            for i, ((text, _), normalized_score) in enumerate(zip(top_results, normalized_scores)):
                # print(f"{i+1}: {text} (Normalized Score: {normalized_score})")
                logger.info(f"{i+1}: {text} (Normalized Score: {normalized_score})")
                ids = [content_id_mapping[text]]  # 获取对应的 id
                # print("Corresponding IDs:")
                logger.info("Corresponding IDs:")
                for doc_id in ids:
                    # print(f"ID: {doc_id}, Content: {text}")
                    logger.info(f"ID: {doc_id}, Content: {text}")
                return_result_ = [item for item in result if item['id'] in ids]
                return_result.extend(return_result_)
            return return_result
        elif entropy > 0.95 or highest_flag == True:
            # max_scores = [score for score in normalized_scores if score == max(normalized_scores)]
            # max_indices = [i for i, score in enumerate(normalized_scores) if score == max(normalized_scores)]
            max_score = max(normalized_scores)
            min_score = max_score - 0.05
            max_indices = [i for i, score in enumerate(normalized_scores) if min_score <= score <= max_score]
            # print("max_indices:", max_indices)  
            logger.info("Highest Score Results:")
            for index in max_indices:
                top_result = top_results[index]
                normalized_score = normalized_scores[index]
                logger.info(f"{top_result[0]} (Normalized Score: {normalized_score})")
                text = top_result[0]
                ids = [content_id_mapping[text]]  # 获取对应的 id
                logger.info("Corresponding IDs:")
                for doc_id in ids:
                    logger.info(f"ID: {doc_id}, Content: {text}")
                return_result_ = [item for item in result if item['id'] in ids]
                return_result.extend(return_result_)
            return return_result
        else:           
            # print("High Score Results:")
            logger.info("High Score Results:")
            high_score_results = [(text, score) for (text, _), score in zip(top_results, normalized_scores) if score>0.5]
            # print("==>:",normalized_scores)
            logger.info(f"==>:{normalized_scores}")
            # print("high_score_results:",high_score_results)
            logger.info(f"high_score_results: {high_score_results}")
            for i, (text, score) in enumerate(high_score_results):
                # print(f"{i+1}: {text} (Normalized Score: {score})")
                logger.info(f"{i+1}: {text} (Normalized Score: {score})")
                ids = [content_id_mapping[text]]  # 获取对应的 id
                # print("Corresponding IDs:")
                logger.info("Corresponding IDs:")
                for doc_id in ids:
                    # print(f"ID: {doc_id}, Content: {text}")
                    logger.info(f"ID: {doc_id}, Content: {text}")
                return_result_ = [item for item in result if item['id'] in ids]
                return_result.extend(return_result_)
            return return_result
            
if __name__ == '__main__':
    save_flag = 0
    if save_flag==1:
        from_filepath="data/在线知识库-QA.xlsx"
        txt_packages = get_packages(from_filepath)
        txt_packages = list(set(txt_packages))
        print("txt_packages_len:",len(txt_packages))
    # search = input(f"请输入要查询的关键字：")

    # recount = int(input(f"请输入要返回的数据条数："))
    embeddings = HuggingFaceEmbeddings(model_name='')
    model_type_path = ""
    embeddings.client = sentence_transformers.SentenceTransformer("/data/staryea/aigc_model/model_chat/bge-large-zh",device="cpu")
    
    if save_flag==1:
        # 批量嵌入txt_packages
        embedded_packages = []
        for package in txt_packages:
            embedding = embeddings.client.encode(package)
            embedded_packages.append(embedding)
        
        # 保存embedded_packages到文件
        np.save("embedded_packages.npy", embedded_packages)
        np.save("txt_packages.npy", txt_packages)
    

    # # 打印嵌入结果
    # for package, embedding in zip(txt_packages, embedded_packages):
    #     print(f"Package: {package}")
    #     print(f"Embedding: {embedding}")
    #     print()
    queries = [ "云电脑办理",
        "关于咪咕商城的国内和国际数据漫游费用，能提供一些详细信息吗？",
               "有没有最新的中国移动5G促销活动或优惠方案？",
               "校园流量10元50G是否提供国内和国际的无限通话套餐选项",
               "麻烦您可不可以介绍一下智享套餐动感校园版套餐？",
"智享套餐动感校园版套餐有什么内容？",
"5G智享套餐 动感校园版 128元 30GB 500分钟 300M宽带 1000小时wlan",
"就问您能不能给我提供云应用的信息",
"我想开通5G新通话功能",
"我想取消云手机业务","找人给我介绍下云手机业务",
"移动云手机",
"帮我介绍一下5G新通话套餐","数字生活5G新通话包","都有哪些咪咕音乐的套餐","我要找5G新通话",
"专线","网络","带宽","宽带","任我选","国漫","流量","5G","手机","校园","套餐","xxx","ffffffff","测试","mmmmmmmmmmm"]
    
    # 加载embedded_packages
    embedded_packages = np.load("embedded_packages.npy")
    txt_packages = np.load("txt_packages.npy")
    
    result = []
    for query in queries:
        search_similar_packages_with_score(query, result, limit=5)
       
    
     

    #/data/staryea/aigc/aigc-server-chat/VECTOR_RECALL
   
    #/data/staryea/aigc/aigc-server-chat/staryea_tools

    #conda activate wenda;cd /data/staryea/aigc/aigc-server-chat

    





    
