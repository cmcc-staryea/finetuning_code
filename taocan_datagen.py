import pandas as pd
import json
import random
import re
import jsonlines
from sklearn.model_selection import train_test_split

num_samples = 1200
key_info_list = ["业务内涵", "产品资费", "业务办理", "业务卖点", "活动对象", "活动卖点", "活动简介", "活动规则"]
map_key_info_dict = {
    "业务内涵": ["业务内涵", "业务的内涵", "内涵"],
    "产品资费": ["产品资费", "资费", "资费标准", "产品的资费", "套餐费用", "套餐的费用","收费"],
    "业务办理": ["业务办理", "业务的办理","办理", "办理流程", "办理方式"],
    "业务卖点": ["业务卖点", "业务的卖点", "特色功能", "优势"],
    "活动对象": ["活动对象", "对象", "活动的对象", "参与对象"],
    "活动卖点": ["活动卖点", "活动的卖点", "吸引点", "特别优惠"],
    "活动简介": ["活动简介", "活动的简介", "活动介绍", "活动的介绍", "活动的内容", "活动内容"],
    "活动规则": ["活动规则","活动的规则", "规则", "参与规则", "参与的规则"]
}

#所有的关键信息对应的词
keyword_list = [value for sublist in map_key_info_dict.values() for value in sublist]
print(keyword_list)

# 每个关键信息对应的标志词
key_info_map = dict()
for key, value in map_key_info_dict.items():
   keywords = list(set(value))
   for kw in keywords:
       key_info_map[kw] = key
print(f"key_info_map:{key_info_map}")

def read_text_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            data.append(line.replace("\n", ""))
    return data


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

def generate_finetuning_dataset(from_filepath,train_filepath,val_filepath):
    # 读取Excel文件并选择问题和intention两列
    df = pd.read_excel(from_filepath)
    df = df[["Q", "A"]]

    # 将数据划分为训练集和验证集
    train_data, val_data = train_test_split(df, test_size=0.001, random_state=42)

    # 将训练集转换为字典格式
    train_formatted_data = []
    train_records = train_data.to_dict('records')
    for item in train_records:
        train_formatted_data.append({"query": item["Q"], "response": item["A"]})

    # 将验证集转换为字典格式
    val_formatted_data = []
    val_records = val_data.to_dict('records')
    for item in val_records:
        val_formatted_data.append({"query": item["Q"], "response": item["A"]})

    # 打乱列表的顺序
    random.shuffle(train_formatted_data)
    # 打乱列表的顺序
    random.shuffle(val_formatted_data)
    # 保存训练集为.jsonl文件
    with jsonlines.open(train_filepath, mode='w') as writer:
        for item in train_formatted_data:
            writer.write(item)

    print("训练集保存成功！")

    # 保存验证集为.jsonl文件
    with jsonlines.open(val_filepath, mode='w') as writer:
        for item in val_formatted_data:
            writer.write(item)

    print("验证集保存成功！")

def find_values(input_question, keyword_list, key_info_map):
    """
    在输入的问题中查找与关键信息列表中的关键信息匹配的值。
    """
    # print("map:",key_info_map)
    # print("keyword_list:",keyword_list)
    found_values = []
    for key_info in keyword_list:
        if key_info in input_question:
            found_values.append(key_info_map[key_info])
    return found_values

def generate_conversation_data(id, system_value, user_value, assistant_value):
    conversation_data = {
        "id": id,
        "conversations": [
            {
                "from": "system",
                "value": system_value
            },
            {
                "from": "user",
                "value": user_value
            },
            {
                "from": "assistant",
                "value": assistant_value
            }
        ]
    }
    return conversation_data

def get_base_question(base_question_template,txt_packages):
    # 创建一个空的DataFrame
    df = pd.DataFrame(columns=['Q', 'A'])

    data = []    
    line_num = 0
  
    for _ in range(num_samples):
        for base_question in base_question_template:
            package_txt = random.choice(txt_packages)
            line_num = line_num + 1
            package_txt = re.sub(r"\s+", "", package_txt) 
            # print(base_question)      
            input_question = base_question.format(package_name=package_txt)
            found_values = find_values(base_question, keyword_list, key_info_map)
            found_values_str = ",".join(list(set(found_values)))
            output_answer = {"套餐名称":package_txt,
                             "关键信息":found_values_str}
            output_answer = json.dumps(output_answer, ensure_ascii=False)
            # print(input_question, " => ",output_answer)
            system_value_str = "你是中国移动（chinamobile）的问答助手。针对用户问话，提取出“套餐名称”及业务关键信息。并以json格式提供，例如：如果套餐名称或关键信息存在，就填上具体内容，如{\"套餐名称\":\"xxx\"，\"关键信息\":\"xxx，xxx，...\"},如果不存在，就置为空字符串，例如：{\"套餐名称\":""，\"关键信息\":""}"
            system_value_str = str(system_value_str)
            # 将输入问题和输出答案添加到DataFrame中
            df = pd.concat([df, pd.DataFrame({'Q': [input_question], 'A': [output_answer]})], ignore_index=True)
            conversation = generate_conversation_data(id=line_num, 
                                                    system_value=system_value_str,
                                                    user_value=input_question,
                                                        assistant_value=output_answer)
            # conversation_str = json.dumps(conversation)
            data.append(conversation)
    return data, df


def get_no_package_question(no_package_template):
    # 创建一个空的DataFrame
    df = pd.DataFrame(columns=['Q', 'A'])

    data = []    
    line_num = 0
    for base_question in no_package_template:
        line_num = line_num + 1
        # print(base_question)  
        found_values = find_values(base_question, keyword_list, key_info_map)
        found_values_str = ",".join(list(set(found_values)))
        output_answer = {"套餐名称":"",
                            "关键信息":found_values_str}
        output_answer = json.dumps(output_answer, ensure_ascii=False)
        # print(input_question, " => ",output_answer)
        system_value_str = "你是中国移动（chinamobile）的问答助手。针对用户问话，提取出“套餐名称”及业务关键信息。并以json格式提供，例如：如果套餐名称或关键信息存在，就填上具体内容，如{\"套餐名称\":\"xxx\"，\"关键信息\":\"xxx，xxx，...\"},如果不存在，就置为空字符串，例如：{\"套餐名称\":""，\"关键信息\":""}"
        system_value_str = str(system_value_str)
        # 将输入问题和输出答案添加到DataFrame中
        df = pd.concat([df, pd.DataFrame({'Q': [base_question], 'A': [output_answer]})], ignore_index=True)
        conversation = generate_conversation_data(id=line_num, 
                                                system_value=system_value_str,
                                                user_value=base_question,
                                                    assistant_value=output_answer)
        # conversation_str = json.dumps(conversation)
        data.append(conversation)
    return data, df

def get_reverse_question(reverse_question_template):
    # 创建一个空的DataFrame
    df = pd.DataFrame(columns=['Q', 'A'])

    data = []    
    line_num = 0
    for reverse_question in reverse_question_template:
        line_num = line_num + 1
        input_question = reverse_question
        output_answer = {"套餐名称":"",
                        "关键信息":""}
        output_answer = json.dumps(output_answer, ensure_ascii=False)
        # print(input_question, " => ",output_answer)
        system_value_str = "你是中国移动（chinamobile）的问答助手。针对用户问话，提取出“套餐名称”及业务关键信息。并以json格式提供，例如：如果套餐名称或关键信息存在，就填上具体内容，如{\"套餐名称\":\"xxx\"，\"关键信息\":\"xxx，xxx，...\"},如果不存在，就置为空字符串，例如：{\"套餐名称\":""，\"关键信息\":""}"
        system_value_str = str(system_value_str)
        # 将输入问题和输出答案添加到DataFrame中
        df = pd.concat([df, pd.DataFrame({'Q': [input_question], 'A': [output_answer]})], ignore_index=True)
        conversation = generate_conversation_data(id=line_num, 
                                                  system_value=system_value_str,
                                                  user_value=input_question,
                                                  assistant_value=output_answer)
        # conversation_str = json.dumps(conversation)
        data.append(conversation)
    
    return data, df


def get_advanced_question(advanced_question_template,txt_packages):
    # 创建一个空的DataFrame
    df = pd.DataFrame(columns=['Q', 'A'])

    location = [
    "北京", "天津", "河北", "山西", "内蒙古",
    "辽宁", "吉林", "黑龙江", "上海", "江苏",
    "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "广西",
    "海南", "重庆", "四川", "贵州", "云南",
    "西藏", "陕西", "甘肃", "青海", "宁夏",
    "新疆", "台湾", "香港", "澳门"
    ]  

    weather = ["天气晴朗","天气多云","天气阴沉","天气寒冷",
               "天气炎热","天气下雨","天气下雪","天气雾霾",
               "晴朗", "晴", "多云", "阴", "雨", "雪", "雾"]
   
    data = []    
    line_num = 0
    
    for _ in range(num_samples):
        for advanced_question in advanced_question_template:
            line_num = line_num + 1

            choice_location = random.choice(location)
            choice_weather = random.choice(weather)
            choice_year = random.randint(2015, 2023)
            choice_package = random.choice(txt_packages)
            # choice_package = choice_package.replace(" ", "")
            choice_package = re.sub(r"\s+", "", choice_package) 
            # print("template",advanced_question)
            # print(choice_location,choice_weather,choice_year,choice_package)

            input_question = advanced_question.replace("{package_name}", str(choice_package)) \
                                            .replace("{year}", str(choice_year)) \
                                            .replace("{location}", str(choice_location)) \
                                            .replace("{weather}", str(choice_weather)) \
                                            
            # print(input_question)                                
           
            found_values = find_values(advanced_question, keyword_list, key_info_map)
            found_values_str = ",".join(list(set(found_values)))
            output_answer = {"套餐名称":choice_package,
                             "关键信息":found_values_str}
            output_answer = json.dumps(output_answer, ensure_ascii=False)
            # print(input_question, " => ",output_answer)
            system_value_str = "你是中国移动（chinamobile）的问答助手。针对用户问话，提取出“套餐名称”及业务关键信息。并以json格式提供，例如：如果套餐名称或关键信息存在，就填上具体内容，如{\"套餐名称\":\"xxx\"，\"关键信息\":\"xxx，xxx，...\"},如果不存在，就置为空字符串，例如：{\"套餐名称\":""，\"关键信息\":""}"
            system_value_str = str(system_value_str)
            # 将输入问题和输出答案添加到DataFrame中
            df = pd.concat([df, pd.DataFrame({'Q': [input_question], 'A': [output_answer]})], ignore_index=True)
            conversation = generate_conversation_data(id=line_num, 
                                                    system_value=system_value_str,
                                                        user_value=input_question,
                                                            assistant_value=output_answer)
            # conversation_str = json.dumps(conversation)
            data.append(conversation)
    return data, df



def write_jsonl_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

if __name__ == '__main__': 
    
    #基础句式   
    file_path = 'data/单套餐问法.txt'  # 替换为实际的文件路径
    base_question_template = read_text_file(file_path)
    # print(base_question_template)
    

    #基础句式   
    file_path = 'data/随机反向增强文本.txt'  # 替换为实际的文件路径
    reverse_question = read_text_file(file_path)
    # print(reverse_question)


    # 进阶句式
    file_path = 'data/进阶单套餐问法.txt'  # 替换为实际的文件路径
    advanced_question_template = read_text_file(file_path)
    
    #无套餐，有关键信息句式
    file_path = 'data/无套餐问法.txt'  # 替换为实际的文件路径
    no_package_question = read_text_file(file_path)


    # base_question_template = base_question_template[:10]
    # advanced_question_template = advanced_question_template[:10]
    # reverse_question = reverse_question[:10]


    data_dir = "data/"
    file_path = data_dir + "/"+"finetuning_package_keyinfo_v3.xlsx"
    train_output_file = data_dir + '/' + "train_package_keyinfo_v3.jsonl"
    val_output_file = data_dir + '/' + "val_package_keyinfo_v3.jsonl"
    conversation_output_file = data_dir + '/' + "conversation_package_keyinfo_v3.jsonl"

    
    from_filepath="data/在线知识库-QA.xlsx"
    txt_packages = get_packages(from_filepath)
    txt_packages = list(set(txt_packages))

    print("get_base_question ... ")
    base_data, base_df = get_base_question(base_question_template, txt_packages)
    
    #无套餐、有关键信息
    print("get_nopackage_question ... ")
    nopackage_data, nopackage_df = get_no_package_question(no_package_question)

    print("get_reverse_question ... ")
    reverse_data, reverse_df = get_reverse_question(reverse_question)
    print("get_advanced_question ... ")
    advanced_data, advanced_df = get_advanced_question(advanced_question_template, txt_packages)
    
    
    random.shuffle(base_data)
    random.shuffle(reverse_data)
    random.shuffle(advanced_data)
    random.shuffle(nopackage_data)

    # 合并并保存data为JSONL格式文件
    merged_data = base_data + advanced_data + reverse_data + nopackage_data
    print("write_jsonl_file ... ")
    write_jsonl_file(conversation_output_file, merged_data)

    # 合并并保存df为Excel文件
    merged_df = pd.concat([base_df, advanced_df, reverse_df, nopackage_df])
    merged_df.to_excel(file_path, index=False)

    total_len = len(base_data) + len(reverse_data) + len(advanced_data) + len(nopackage_data)
    # Print file write status and additional information
    print("Conversation data saved to:", conversation_output_file)
    print("Total number of conversations:", total_len)
 
    print("generate_finetuning_dataset ... ")
    generate_finetuning_dataset(file_path,train_output_file,val_output_file)
    
    print("Data written successfully.")
    
    
    
