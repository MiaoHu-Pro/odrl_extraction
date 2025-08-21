import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from importlib import import_module
import numpy as np

# 定义BERT特殊符号和类别映射
CLS = '[CLS]'
# id_to_name = {0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education', 4: 'science',
#               5: 'society', 6: 'politics', 7: 'sports', 8: 'game', 9: 'entertainment'}

id_to_name = {0: 'non', 1: 'odrl'}
# 加载BERT情感分析模型和相关配置
model_name = 'bert'
x = import_module('models.' + model_name)
config = x.Config()
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True

model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))

# 推理函数，用于对输入文本进行分类分析,与模型预测部分是一样的
def inference(model, config, input_text, pad_size=32):
    # 对输入文本进行分词和预处理
    content = config.tokenizer.tokenize(input_text)
    content = [CLS] + content
    seq_len = len(content)
    token_ids = config.tokenizer.convert_tokens_to_ids(content)
    # 填充或截断文本到指定长度
    if seq_len < pad_size:
        mask = [1] * len(token_ids) + [0] * (pad_size - seq_len)
        token_ids += [0] * (pad_size - seq_len)
    else:
        mask = [1] * pad_size
        token_ids = token_ids[:pad_size]
        seq_len = pad_size
    # 将处理后的文本转换为Tensor形式
    x = torch.LongTensor(token_ids).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)
    mask = torch.LongTensor(mask).to(config.device)
    # 增加一个维度
    x = x.unsqueeze(0)
    seq_len = seq_len.unsqueeze(0)
    mask = mask.unsqueeze(0)
    data = (x, seq_len, mask)
    output = model(data)
    # 获取预测结果并返回
    predict_result = torch.max(output.data, 1)[1]
    predict_result = predict_result.item()
    predict_result = id_to_name[predict_result]

    return predict_result

# 创建Flask应用
# Initialize Flask
app = Flask(__name__)
CORS(app)

# Serve the HTML page
@app.route('/')
def index():
    return render_template('prediction.html')

# 定义路由，接收POST请求并进行推理
# @app.route('/prediction/', methods=["POST"])
# def main_server():
#     # 从POST请求中获取用户ID和文本数据
#     uid = request.form['uid']
#     text = request.form['text']
#     # 调用推理函数获取预测结果
#     res = inference(model, config, text)
#     return jsonify({"uid": uid, "category": res})
#     # return res

@app.route('/prediction/', methods=["POST"])
def main_server():
    uid = request.form.get('uid')
    text = request.form.get('text')

    if not uid or not text:
        return jsonify({"error": "Missing uid or text"}), 400

    try:
        res = inference(model, config, text)
        return jsonify({"uid": uid, "text":text, "category": res})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 如果脚本作为主程序运行，则启动Flask应用
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
