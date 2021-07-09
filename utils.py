import collections
import json
import pickle
import re

import javalang
import torch
import numpy as np
from tqdm import tqdm
# from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
#
# from IR.bert_whitening import sents_to_vecs, transform_and_normalize

COMMENT_RX = re.compile("(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/", re.MULTILINE)
SPACE_RX = re.compile('[\n\r$\s]+', re.MULTILINE)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_source(code):
    code = code.replace('\n',' ').strip()
    tokens = list(javalang.tokenizer.tokenize(code))
    tks = []
    for tk in tokens:
        if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
            tks.append('STR_')
        elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
            tks.append('NUM_')
        elif tk.__class__.__name__ == 'Boolean':
            tks.append('BOOL_')
        else:
            tks.append(tk.value)
    return " ".join(tks)

def get_ast(processed_code):
    code = processed_code.strip()
    tokens = javalang.tokenizer.tokenize(code)
    token_list = list(javalang.tokenizer.tokenize(code))
    length = len(token_list)
    parser = javalang.parser.Parser(tokens)
    try:
        tree = parser.parse_member_declaration()
    except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
        print(code)
    flatten = []
    for path, node in tree:
        flatten.append({'path': path, 'node': node})
    ign = False
    outputs = []
    stop = False
    for i, Node in enumerate(flatten):
        d = collections.OrderedDict()
        path = Node['path']
        node = Node['node']
        children = []
        for child in node.children:
            child_path = None
            if isinstance(child, javalang.ast.Node):
                child_path = path + tuple((node,))
                for j in range(i + 1, len(flatten)):
                    if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                        children.append(j)
            if isinstance(child, list) and child:
                child_path = path + (node, child)
                for j in range(i + 1, len(flatten)):
                    if child_path == flatten[j]['path']:
                        children.append(j)
        d["id"] = i
        n = str(node)
        n = n[:n.find('(')]
        d["type"] = n
        if children:
            d["children"] = children
        value = None
        if hasattr(node, 'name'):
            value = node.name
        elif hasattr(node, 'value'):
            value = node.value
        elif hasattr(node, 'position') and node.position:
            for i, token in enumerate(token_list):
                if node.position == token.position:
                    pos = i + 1
                    value = str(token.value)
                    while (pos < length and token_list[pos].value == '.'):
                        value = value + '.' + token_list[pos + 1].value
                        pos += 2
                    break
        elif type(node) is javalang.tree.This \
                or type(node) is javalang.tree.ExplicitConstructorInvocation:
            value = 'this'
        elif type(node) is javalang.tree.BreakStatement:
            value = 'break'
        elif type(node) is javalang.tree.ContinueStatement:
            value = 'continue'
        elif type(node) is javalang.tree.TypeArgument:
            value = str(node.pattern_type)
        elif type(node) is javalang.tree.SuperMethodInvocation \
                or type(node) is javalang.tree.SuperMemberReference:
            value = 'super.' + str(node.member)
        elif type(node) is javalang.tree.Statement \
                or type(node) is javalang.tree.BlockStatement \
                or type(node) is javalang.tree.ForControl \
                or type(node) is javalang.tree.ArrayInitializer \
                or type(node) is javalang.tree.SwitchStatementCase:
            value = 'None'
        elif type(node) is javalang.tree.VoidClassReference:
            value = 'void.class'
        elif type(node) is javalang.tree.SuperConstructorInvocation:
            value = 'super'

        if value is not None and type(value) is type('str'):
            d['value'] = value
        if not children and not value:
            # print('Leaf has no value!')
            print(type(node))
            print(code)
            ign = True
            # break
        outputs.append(d)
    if not ign:
        return json.dumps(outputs)

def AST_(cur_root_id, node_list):
    cur_root = node_list[cur_root_id]
    list = []
    tmp_dict = {}
    tmp_dict['id'] = cur_root['id']
    tmp_dict['type'] = cur_root['type']

    if 'children' in cur_root:
        chs = cur_root['children']
        temp_list = []
        for ch in chs:
            temp_list.append(AST_(ch, node_list))
        tmp_dict['children'] = temp_list
    if 'value' in cur_root:
        tmp_dict['name'] = cur_root['value']
        return tmp_dict
    list.append(tmp_dict)
    return list

def SBT_(cur_root_id, node_list):
    cur_root = node_list[cur_root_id]
    tmp_list = []
    # tmp_list.append("(")

    str = cur_root['type']
    tmp_list.append(str)

    if 'children' in cur_root:
        chs = cur_root['children']
        for ch in chs:
            tmp_list.extend(SBT_(ch, node_list))
    # tmp_list.append(")")
    # tmp_list.append(str)
    return tmp_list

def get_sbt_structure(ast):
    ast = json.loads(ast)
    ast_sbt = SBT_(0, ast)
    return ' '.join(ast_sbt)

def get_ast_structure(ast):
    ast = json.loads(ast)
    ast = AST_(0, ast)
    return ast

def hump2underline(hunp_str):
    '''
    驼峰形式字符串转成下划线形式
    :param hunp_str: 驼峰形式字符串
    :return: 字母全小写的下划线形式字符串
    '''
    # 匹配正则，匹配小写字母和大写字母的分界位置
    p = re.compile(r'([a-z]|\d)([A-Z])')
    # 这里第二个参数使用了正则分组的后向引用
    sub = re.sub(p, r'\1 \2', hunp_str).lower()
    return sub

def get_func_name(code):
    func_name = ""
    tokens = javalang.tokenizer.tokenize(code)
    token_list = list(javalang.tokenizer.tokenize(code))
    length = len(token_list)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    flatten = []
    for path, node in tree:
        flatten.append({'path': path, 'node': node})
    outputs = []
    for i, Node in enumerate(flatten):
        d = collections.OrderedDict()
        path = Node['path']
        node = Node['node']
        children = []
        for child in node.children:
            child_path = None
            if isinstance(child, javalang.ast.Node):
                child_path = path + tuple((node,))
                for j in range(i + 1, len(flatten)):
                    if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                        children.append(j)
            if isinstance(child, list) and child:
                child_path = path + (node, child)
                for j in range(i + 1, len(flatten)):
                    if child_path == flatten[j]['path']:
                        children.append(j)
        d["id"] = i
        n = str(node)
        n = n[:n.find('(')]
        d["type"] = n
        if children:
            d["children"] = children
        value = None
        if hasattr(node, 'name'):
            value = node.name
        elif hasattr(node, 'value'):
            value = node.value
        elif hasattr(node, 'position') and node.position:
            for i, token in enumerate(token_list):
                if node.position == token.position:
                    pos = i + 1
                    value = str(token.value)
                    while (pos < length and token_list[pos].value == '.'):
                        value = value + '.' + token_list[pos + 1].value
                        pos += 2
                    break
        elif type(node) is javalang.tree.This \
                or type(node) is javalang.tree.ExplicitConstructorInvocation:
            value = 'this'
        elif type(node) is javalang.tree.BreakStatement:
            value = 'break'
        elif type(node) is javalang.tree.ContinueStatement:
            value = 'continue'
        elif type(node) is javalang.tree.TypeArgument:
            value = str(node.pattern_type)
        elif type(node) is javalang.tree.SuperMethodInvocation \
                or type(node) is javalang.tree.SuperMemberReference:
            value = 'super.' + str(node.member)
        elif type(node) is javalang.tree.Statement \
                or type(node) is javalang.tree.BlockStatement \
                or type(node) is javalang.tree.ForControl \
                or type(node) is javalang.tree.ArrayInitializer \
                or type(node) is javalang.tree.SwitchStatementCase:
            value = 'None'
        elif type(node) is javalang.tree.VoidClassReference:
            value = 'void.class'
        elif type(node) is javalang.tree.SuperConstructorInvocation:
            value = 'super'

        if value is not None and type(value) is type('str'):
            d['value'] = value
        if not children and not value:
            # print('Leaf has no value!')
            print(type(node))
            print(code)
            ign = True
            # break
        outputs.append(d)
    for d in outputs:
        if(d['type']=="MethodDeclaration"):
            func_name = d['value']
    func_name = func_name.strip()
    return func_name

def get_func_body(code, func_name):
    index = code.find(func_name)
    code = code[index:]
    index_2 = code.find('{')
    func_body = code[index_2 : -1]
    func_body = func_body.replace("\n","").strip()
    tokens = list(javalang.tokenizer.tokenize(func_body))
    tks = []
    for tk in tokens:
        if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
            tks.append('STR_')
        elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
            tks.append('NUM_')
        elif tk.__class__.__name__ == 'Boolean':
            tks.append('BOOL_')
        else:
            tks.append(tk.value)
    return " ".join(tks)
#
# def softmax(x):
#     x_exp = np.exp(x)
#     x_sum = np.sum(x_exp, axis = 0, keepdims = True)
#     s = x_exp / x_sum
#     return s.tolist()
#
# # 初始化BERT 预训练的模型
# def predict(text_a, text_b):
#     tokenizer = RobertaTokenizer.from_pretrained('./model/matchcode')
#     model = RobertaForSequenceClassification.from_pretrained("./model/matchcode")  # BERT 配置文件
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(DEVICE)
#     tokens_pt2 = tokenizer(
#         text=text_a,
#         text_pair=text_b,
#         truncation=True,
#         padding="max_length",
#         max_length=256,
#         return_tensors="pt",
#     )
#     # tokens_pt2 = tokenizer.encode_plus(text, return_tensors="pt")
#     input_ids = tokens_pt2['input_ids']
#     attention_mask = tokens_pt2['attention_mask']
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids.to(DEVICE),
#                         attention_mask=attention_mask.to(DEVICE))['logits']
#         logit = outputs[0].cpu().numpy().tolist()
#         logit = softmax(logit)
#         _, y = torch.max(outputs, dim=1)
#     # return logit,y.item()
#     return y.item()
#

def transformer(code):
    code = COMMENT_RX.sub('', code)
    code = SPACE_RX.sub(' ', code)
    processed_code = process_source(code)
    code_seq = ' '.join([hump2underline(i) for i in processed_code.split()])
    ast = get_ast(processed_code)
    sbt = get_sbt_structure(ast)
    return code_seq, sbt
#
# f=open('IR/func_body_vector_whitening.pkl', 'rb')
# vec=pickle.load(f)
# f.close()
#
# f=open('IR/kernel.pkl', 'rb')
# kernel=pickle.load(f)
# f.close()
#
# f=open('IR/bias.pkl', 'rb')
# bias=pickle.load(f)
# f.close()
#
# import pandas as pd
#
# def sim_jaccard(s1, s2):
#     """jaccard相似度"""
#     s1, s2 = set(s1), set(s2)
#     ret1 = s1.intersection(s2)  # 交集
#     ret2 = s1.union(s2)  # 并集
#     sim = 1.0 * len(ret1) / len(ret2)
#     return sim
#
# def repair(body):
#     df = pd.read_csv("./IR/right_clean.csv")
#     func_body, func_name = df['func_body'].tolist(), df['func_name'].tolist()
#     #
#     #
#     # tokenizer = RobertaTokenizer.from_pretrained("./model/codebert")
#     # model = RobertaModel.from_pretrained("./model/codebert")
#     # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # model.to(DEVICE)
#     # body = sents_to_vecs([body], tokenizer, model)
#     # body = transform_and_normalize(body, kernel, bias)
#     result_list = []
#     for i in tqdm(range(len(func_body))):
#         score = sim_jaccard(func_body[i].split(), body.split())
#         result_list.append((score, i))
#
#     result_list.sort(reverse=True)
#     result = result_list[0]
#     suggest_name = func_name[result[1]].split()
#     temp_list = []
#     for j in range(len(suggest_name)):
#         if(j>0):
#             suggest_name[j] = suggest_name[j].capitalize()
#         temp_list.append(suggest_name[j])
#     suggest_name = "".join(temp_list)
#     return suggest_name