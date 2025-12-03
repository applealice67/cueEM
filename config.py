from transformers import BertTokenizer, BertModel

import logging
from datetime import datetime
from transformers import BertTokenizer, BertModel
import torch.optim as optim
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import csv

tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
model = BertModel.from_pretrained('./bert-base-uncased')

path = 'Structured'
# path='data/er_magellan/Dirty'
# path='data/er_magellan/Textual'

# dataset="Amazon-Google"
dataset="Beer"
# dataset="Fodors-Zagats"
# dataset="iTunes-Amazon"
#dataset = "Walmart-Amazon"
# dataset="DBLP-ACM"
# dataset="DBLP-GoogleScholar"
# dataset="Abt-Buy"


max_len = 64
num_epochs = 10
lr = 3e-5
lambda_l2 = 1e-4
# lambda_l2 = 0
# dropout = 0.3
batch_size = 16

# {'title': 0.14652534375208545, 'manufacturer': 0.05769829852974523, 'price': 0.20776585468018058,
if dataset == 'Amazon-Google':  # one epoch is 2 hour
    attrWeight = [0.8, 0.2, 0.8, 0.5]

# {'Beer_Name': 0.24430906805413774, 'Brew_Factory_Name': 0.28038944996494647, 'Style': -0.27087976005199366, 'ABV': 0.20007473887936267,
elif dataset == 'Beer':  # one epoch is 8min
    # self.weights = nn.Parameter(torch.tensor([0.254, 0.257, 0.0, 0.209, 0.28]))
    attrWeight = [0.8, 0.8, 0.1, 0.8, 0.5]
    # batch_size = 32

# {'name': 0.47258647660820996, 'addr': 0.25200868914159175, 'city': 0.0546193907431387, 'phone': 0.3766036095312342, 'type': 0.019489420037281442, 'class': 0.5555150228810763,
elif dataset == 'Fodors-Zagats':  # one epoch is 8min
    max_len = 128
    num_epochs = 5
    attrWeight = [0.8, 0.8, 0.2, 0.8, 0.2, 0.8, 0.5]

# {'Song_Name': 0.44707260216969646, 'Artist_Name': -0.004871933921785437, 'Album_Name': 0.2745449893100449, 'Genre': -0.17923460338106267, 'Price': 0.12493290395717868, 'CopyRight': 0.0294511173021221, 'Time': 0.6669707188995297, 'Released': -0.05005850366590336,
elif dataset == 'iTunes-Amazon':  # one epoch is 15min
    attrWeight = [0.8, 0.1, 0.8, 0.1, 0.8, 0.1, 0.8, 0.1, 0.5]

# {'title': -0.01971571078298578, 'category': 0.045308931761990134, 'brand': -0.02751322714297252, 'modelno': 0.1398147006881398, 'price': 0.15020384295658573}
elif dataset == 'Walmart-Amazon':  # one epoch is 1h50min
    attrWeight = [0.1, 0.2, 0.1, 0.8, 0.8, 0.5]

# {'title': np.float64(0.6220293978868603), 'authors': np.float64(0.5413259716183666), 'venue': np.float64(0.18501210507981883), 'year': np.float64(0.5792473297934675), 'overall': np.float64(0.5064537462523807)}
elif dataset == 'DBLP-ACM':  # one epoch is 1h50min
    attrWeight = [0.8, 0.8, 0.8, 0.8, 0.5]

# {'title': np.float64(0.3845101758931991), 'authors': np.float64(0.44533613198280725), 'venue': np.float64(0.21294559276127886), 'year': np.float64(0.06534656937611615), 'overall': np.float64(0.3845101758931991)}
elif dataset == 'DBLP-GoogleScholar':  # one epoch is 1h50min
    attrWeight = [0.7, 0.7, 0.7, 0.2, 0.5]

# {'name': np.float64(0.12580924334638566), 'description': np.float64(-0.08685249301028268), 'price': np.float64(0.04595287378569339), 'overall': np.float64(0.12580924334638566)}
elif dataset == 'Abt-Buy':  # one epoch is 1h50min
    attrWeight = [0.8, 0.1, 0.2, 0.5]

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 格式化时间戳
logfile = f"{dataset}_{current_time}.log"
logging.basicConfig(
    filename=logfile,  # 指定日志文件名
    filemode='w',  # 写入模式（覆盖旧文件）
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import re

number_pattern = re.compile(r'\d+(\.\d+)?')


def extract_number(s):
    match = number_pattern.search(s)
    return float(match.group()) if match else 0


def subtract_numbers(str1, str2):
    num1 = extract_number(str1)
    num2 = extract_number(str2)

    result = int(abs(num1 - num2))

    return result


def replace_template(template, row):
    s1 = template
    s2 = template

    for key in row:
        if key == "label":
            continue
        t1, t2 = row[key]
        str = '[' + key + ']'
        if not s1:
            s1 = s1 + " " + t1
        else:
            s1 = s1.replace(str, t1)
        if not s2:
            s2 = s2 + " " + t2
        else:
            s2 = s2.replace(str, t2)
    return s1, s2


def get_text(row):
    if dataset == 'Amazon-Google':
        template = "[title] by [manufacturer] now only [price]"


    elif dataset == 'Beer':
        template = "[Beer_Name] crafted by [Brew_Factory_Name] is a [Style] beer with [ABV]"
        # template = "[Beer_Name] crafted by [Brew_Factory_Name] is a beer with [ABV]"


    elif dataset == 'Fodors-Zagats':
        template = "[name] from [class] [addr] [city] is [type] and phone is [phone]"
        # template = "[name] from [class] in [addr] is [type] and phone is [phone]"

    elif dataset == 'iTunes-Amazon':
        template = "[Song_Name] by [Artist_Name] from [Album_Name] in [Genre] released [Released] [CopyRight] now only [Price] with Duration [Time]"
        # template = "[Song_Name] by [Artist_Name] from [CopyRight] [Album_Name] released [Released] now only [Price] with [Time]"

    elif dataset == 'Walmart-Amazon':
        # template = "[title] from [brand] [category] [modelno] now only [price]"
        template = "[title] from [brand] [category] [modelno] now only [price]"

    elif dataset == 'DBLP-ACM':
        template = "[title] by [authors] at [venue] in [year]"

    elif dataset == 'DBLP-GoogleScholar':
        template = "[title] by [authors] at [venue] in [year]"

    elif dataset == 'Abt-Buy':
        template = "[name] with [description] now only [price]"
    else:
        template = ""

    s1, s2 = replace_template(template, row)
    return s1, s2


# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

def noStopwords(line):
    line = re.sub(r'[^\w\s]', '', line)
    stopwords = set(
        ['such', "you'd", 'y', 't', 'down', 'i', 'by', 'whom', 'most', 'his', 'does', 'are', 'between', 're', 'isn',
         'only', 'she', 'of', 'had', 'through', 'other', 'needn', 'be', 'below', 'should', 'when', 'on', 'for', "don't",
         'until', 'can', 'to', 'a', 'from', 'has', "you'll", 'few', 'were', "that'll", 'while', 'just', "she's",
         "didn't", 'again', 'under', 'him', 'these', 'your', 'this', 'that', 'being', 'doing', 'all', 'with', "haven't",
         'didn', 'nor', 'they', 'where', 'our', 'them', 'couldn', 'm', "needn't", 'me', 'you', 'we', 'than', "wouldn't",
         "shan't", 'ma', 'won', 'yourselves', 'wouldn', 'haven', "it's", 'against', 'ain', 'have', 's', 'any', 'do',
         'himself', 'there', 'what', 'myself', 'both', 've', 'up', 'mustn', 'or', 'wasn', 'into', 'which', "shouldn't",
         'hadn', 'as', 'own', 'o', 'mightn', 'an', 'don', 'her', 'weren', 'itself', 'those', 'how', 'hers', "mightn't",
         'is', 'was', "wasn't", 'before', 'if', 'it', 'will', 'once', 'did', 'same', "hadn't", 'now', 'll', 'no',
         'shan', "you're", 'too', 'aren', 'he', 'some', 'my', 'over', "doesn't", 'shouldn', "isn't", 'ourselves', 'd',
         'am', 'themselves', "aren't", 'off', 'having', 'in', "hasn't", 'further', "mustn't", 'yourself', 'ours',
         'theirs', 'here', 'more', 'so', "won't", 'very', "should've", 'out', 'the', 'and', 'who', 'their', 'but',
         "couldn't", 'hasn', 'doesn', 'not', 'above', 'because', 'about', 'its', 'during', "weren't", 'herself', 'been',
         'yours', "you've", 'why', 'after', 'then', 'each', 'at'])
    tokens = line.split()
    filtered_tokens = [w.lower() for w in tokens if w.lower() not in stopwords]
    line = ' '.join(filtered_tokens)
    return line


def parse_line(line):
    """解析单行字符串"""

    result = {}

    tokens = line.split()

    key = None
    value = []

    for token in tokens:
        if token == "COL":
            if key:
                # 保存当前字段名和值
                if key in result:
                    if isinstance(result[key], tuple):
                        result[key] += (" ".join(value),)
                    else:
                        result[key] = (result[key], " ".join(value))
                else:
                    result[key] = " ".join(value)
            key = None
            value = []
        elif token == "VAL":
            key = key  # 确保 key 已经被赋值
        elif key is None:
            key = token  # 当前 token 是字段名
        else:
            value.append(token)  # 当前 token 是值的一部分

    # 处理最后一个字段
    if key:
        if key in result:
            if isinstance(result[key], tuple):
                result[key] += (" ".join(value),)
            else:
                result[key] = (result[key], " ".join(value))
        else:
            result[key] = " ".join(value)

    # s1,s2=get_text(result)
    # result["overall"]=(s1,s2)
    # 检查最后一个数字是否为 label
    if value and value[-1].isdigit():
        result["label"] = int(value[-1])
        value.pop()  # 移除最后一个数字
        # 更新最后一个字段的值
        if key in result:
            if isinstance(result[key], tuple):
                result[key] = result[key][:-1] + (" ".join(value),)
            else:
                result[key] = " ".join(value)

    label = result["label"]
    del result['label']
    s1, s2 = get_text(result)
    result["overall"] = (s1, s2)
    result["label"] = label

    return result


def parse_file(file_path):
    """从文件中读取每一行并解析"""
    final_result = {}
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()  # 去掉行首行尾的空白字符
        if not line:
            continue  # 跳过空行
        parsed_line = parse_line(line)

        # 将解析结果合并到最终字典
        for key, value in parsed_line.items():

            if key == 'overall':
                value = (noStopwords(value[0]), noStopwords(value[1]))

            if key in final_result:
                final_result[key].append(value)
            else:
                final_result[key] = [value]
    key = ''
    if "price" in final_result:
        key = "price"
    elif "ABV" in final_result:
        key = 'ABV'
    elif "Price" in final_result:
        key = 'Price'
    elif "year" in final_result:
        key = 'year'
    if key:
        processed_results = []
        for value_tuple in final_result[key]:
            if isinstance(value_tuple, tuple):
                diff = subtract_numbers(value_tuple[0], value_tuple[1])
                processed_results.append((str(0), str(diff)))  # 更改为 (0, 绝对值整数) 的形式
            else:
                processed_results.append((str(0), str(0)))  # 单个值或无效值
        final_result[key] = processed_results

    # keys_list = list(final_result.keys())
    # for i in range(len(attrWeight)):
    #     if attrWeight[i]< 0.1:
    #         del final_result[keys_list[i]]

    return final_result


def dict_list(dic_data):
    keys = list(dic_data.keys())
    print(keys)
    logging.info(keys)
    list_data = []
    for i in range(len(dic_data["label"])):
        item = []
        for key in keys:
            item.append(dic_data[key][i])
        list_data.append(item)
    return list_data


def data_augment(list_data):
    result_data = []
    t_data = []
    f_data = []
    for row in list_data:
        if row[-1]:
            t_data.append(row)
        else:
            f_data.append(row)
    len_t = len(t_data)
    len_f = len(f_data)
    len_max = max(len_t, len_f)
    print(f"t_len:{len_t},f_len:{len_f}")
    logging.info(f"t_len:{len_t},f_len:{len_f}")

    for i in range(len_max):
        result_data.append(t_data[i % len_t])
        result_data.append(f_data[i % len_f])
    return result_data


def get_data(dataset):
    train_path = f'./{path}/{dataset}/train.txt'
    train_data = data_augment(dict_list(parse_file(train_path)))
    test_path = f'./{path}/{dataset}/test.txt'
    test_data = dict_list(parse_file(test_path))
    pairnums = len(train_data[0]) - 1
    print(f"pair nums is {pairnums}".format(pairnums))
    logging.info(f"pair nums is {pairnums}".format(pairnums))
    train_path = f'./{path}/{dataset}/train.csv'
    with open(train_path, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        for data in train_data:
            csv_writer.writerow(data)
    test_path = f'./{path}/{dataset}/test.csv'
    with open(test_path, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        for data in test_data:
            csv_writer.writerow(data)

    return train_data, test_data, pairnums


trainData, testData, pairNum = get_data(dataset)
print("load trainData,testData")

result_path = f'./{path}/{dataset}/result.csv'
save_path = f'./{path}/{dataset}/best_model.pth'


class WeightedClassifier(nn.Module):
    def __init__(self, hidden_size, num_chunks):
        super(WeightedClassifier, self).__init__()
        self.num_chunks = num_chunks  # 动态传入分割数量
        self.linear = nn.Linear(hidden_size * num_chunks, 1)  # 线性层的输入维度根据 num_chunks 动态调整
        # self.weights = nn.Parameter(torch.ones(num_chunks))  # 初始化权重为可学习参数，数量根据 num_chunks 动态调整
        self.weights = nn.Parameter(torch.tensor(attrWeight))

    def forward(self, x):
        # x的形状为 [batch_size, hidden_size * num_chunks]
        # 将x分成 num_chunks 部分，分别对应不同特征
        x_chunks = x.chunk(self.num_chunks, dim=1)

        # 将权重应用于特征
        weighted_chunks = [chunk * self.weights[i] for i, chunk in enumerate(x_chunks)]

        # 拼接加权后的特征
        weighted_x = torch.cat(weighted_chunks, dim=1)

        # 通过线性层
        logits = self.linear(weighted_x)
        return logits


classifier = WeightedClassifier(model.config.hidden_size, pairNum)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr, weight_decay=lambda_l2)
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)


from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch


class SentenceSimilarityDataset(Dataset):
    def __init__(self, data, max_len):
        """
        初始化函数。
        :param data: 输入数据，每个元素是一个列表，其中包含若干个文本对（元组）和一个标签（数值）。
        :param max_len: 每个句子的最大长度。
        """
        self.max_length = max_len
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 提取当前索引的文本对列表和标签
        label = self.data[idx][-1]  # 标签是列表的最后一个元素
        sentence_list = self.data[idx][:-1]  # 文本对列表是除了最后一个元素之外的部分

        encodings_list = []
        for text1, text2 in sentence_list:
            encodings = self.tokenizer(
                text1,
                text2,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
                add_special_tokens=True
            )
            encodings_list.append(encodings)

        # 提取编码结果
        input_ids = [enc['input_ids'].flatten() for enc in encodings_list]
        token_type_ids = [enc['token_type_ids'].flatten() for enc in encodings_list]
        attention_mask = [enc['attention_mask'].flatten() for enc in encodings_list]

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train(trainData, testData):
    train_dataset = SentenceSimilarityDataset([data for data in trainData], max_len)
    test_dataset = SentenceSimilarityDataset([data for data in testData], max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_f1 = 0
    best_epoch = 0
    best_model_state = None

    for epoch in range(num_epochs):
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        print("current time:", formatted_time)

        model.train()  # 切换到训练模式
        train_loss = 0.0
        train_steps = 0

        for batch in train_loader:

            # labels = batch['labels'].view(-1, 1).float()  # 确保labels是float类型
            # change
            input_ids_list = [batch['input_ids'][i] for i in range(pairNum)]
            attention_mask_list = [batch['attention_mask'][i] for i in range(pairNum)]
            labels = batch['labels'].view(-1, 1).float()
            outputs_list = []
            for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask)
                outputs_list.append(outputs.last_hidden_state[:, 0, :])
                print(str(outputs_list))

            combined_outputs = torch.cat(outputs_list, dim=1)

            logits = classifier(combined_outputs)

            loss = loss_fn(logits, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 累积训练损失
            train_loss += loss.item() * input_ids.size(0)
            train_steps += 1
            print(train_steps)

        train_loss = train_loss / train_steps
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}")
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}")

        # 评估模型
        model.eval()  # 切换到评估模式
        test_predictions = []
        test_labels_list = []

        with torch.no_grad():  # 在评估过程中不计算梯度
            for batch in test_loader:
                # change
                input_ids_list = [batch['input_ids'][i] for i in range(pairNum)]
                attention_mask_list = [batch['attention_mask'][i] for i in range(pairNum)]
                labels = batch['labels']
                outputs_list = []
                for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                    outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask)
                    outputs_list.append(outputs.last_hidden_state[:, 0, :])
                combined_outputs = torch.cat(outputs_list, dim=1)
                # combined_outputs = sum(outputs_list)
                logits = classifier(combined_outputs)

                # logits = classifier(outputs.last_hidden_state.mean(dim=1))
                # logits = classifier(outputs.last_hidden_state.max(dim=1).values)
                predicted_labels = torch.sigmoid(logits).round()  # 将logits转换为0或1

                test_predictions.extend(predicted_labels.tolist())
                test_labels_list.extend(labels.tolist())

        # 计算准确率、召回率和F1分数
        accuracy = accuracy_score(test_labels_list, test_predictions)
        recall = recall_score(test_labels_list, test_predictions)
        precision = precision_score(test_labels_list, test_predictions)
        f1 = f1_score(test_labels_list, test_predictions)
        print(f"Test Accuracy: {accuracy:.4f}, precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        logging.info(
            f"Test Accuracy: {accuracy:.4f}, precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            # 保存当前最佳模型状态
            best_model_state = model.state_dict()
            # torch.save(best_model_state, 'best_model_state.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
            }, save_path)

            print(f"best from epoch {best_epoch + 1} with best F1 Score: {best_f1:.4f}")
            logging.info(f"best from epoch {best_epoch + 1} with best F1 Score: {best_f1:.4f}")

        else:
            # 如果F1没有提高，则加载之前的最佳模型状态并继续训练
            if best_model_state:
                # model.load_state_dict(best_model_state)
                # 加载模型和分类器
                checkpoint = torch.load(save_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                classifier.load_state_dict(checkpoint['classifier_state_dict'])

                # 查看加载后的权重
                learned_weights = classifier.weights.data
                print(f"Loaded weights for sentence pairs:{learned_weights}")
                logging.info(f"Loaded weights for sentence pairs:{learned_weights}")
                print(f"Loading model state from epoch {best_epoch + 1} with best F1 Score: {best_f1:.4f}")
                logging.info(f"Loading model state from epoch {best_epoch + 1} with best F1 Score: {best_f1:.4f}")


def eval(test_data, result_file):
    # 加载保存的模型状态
    # model_state_dict = torch.load('best_model_state.pth')  # 根据需要调整map_location
    # model.load_state_dict(model_state_dict)

    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    # 查看加载后的权重
    learned_weights = classifier.weights.data
    print(f"Loaded weights for sentence pairs:{learned_weights}")
    logging.info(f"Loaded weights for sentence pairs:{learned_weights}")

    model.eval()
    # change
    test_dataset = SentenceSimilarityDataset([data for data in test_data], max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_predictions = []
    test_labels_list = []

    with torch.no_grad():  # 在评估过程中不计算梯度
        for batch in test_loader:
            # change
            input_ids_list = [batch['input_ids'][i] for i in range(pairNum)]
            attention_mask_list = [batch['attention_mask'][i] for i in range(pairNum)]
            labels = batch['labels']
            outputs_list = []
            for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask)
                outputs_list.append(outputs.last_hidden_state[:, 0, :])
            combined_outputs = torch.cat(outputs_list, dim=1)
            # combined_outputs = sum(outputs_list)
            logits = classifier(combined_outputs)
            # logits = classifier(outputs.last_hidden_state.mean(dim=1))
            # logits = classifier(outputs.last_hidden_state.max(dim=1).values)
            predicted_labels = torch.sigmoid(logits).round()  # 将logits转换为0或1
            test_predictions.extend(predicted_labels.tolist())
            test_labels_list.extend(labels.tolist())

    # 计算准确率、召回率和F1分数
    accuracy = accuracy_score(test_labels_list, test_predictions)
    recall = recall_score(test_labels_list, test_predictions)
    precision = precision_score(test_labels_list, test_predictions)
    f1 = f1_score(test_labels_list, test_predictions)
    print(f"Test Accuracy: {accuracy:.4f}, precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    logging.info(f"Test Accuracy: {accuracy:.4f}, precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    with open(result_file, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["id", "lable", "predict"])
        for i in range(len(test_labels_list)):
            data = []
            data.append(i)
            data.append(test_labels_list[i])
            data.append(test_predictions[i])
            csv_writer.writerow(data)


train(trainData, testData)
eval(testData, result_path)