import os
import json
import random
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import paddle
import paddlenlp
import paddle.nn.functional as F
from functools import partial
from paddlenlp.data import Stack, Dict, Pad
from paddlenlp.datasets import load_dataset
import paddle.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from paddlenlp.transformers.auto.tokenizer import AutoTokenizer
from paddlenlp.transformers.ernie.modeling import ErniePretrainedModel,ErnieForSequenceClassification

seed = 2022
paddle.seed(seed)
random.seed(seed)
np.random.seed(seed)

# 超参数
MODEL_NAME = 'ernie-3.0-base-zh'
# 设置最大阶段长度 和 batch_size
max_seq_length = 365
train_batch_size = 8
valid_batch_size = 8
test_batch_size = 8
# 训练过程中的最大学习率
learning_rate = 8e-5
# 训练轮次
epochs = 50
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01
max_grad_norm = 1.0
# 提交文件名称
sumbit_name = "work/sumbit.csv"
model_logging_dir = 'work/model_logging.csv'
early_stopping = 10
# Rdrop Loss的超参数，若该值大于0.则加权使用R-drop loss
rdrop_coef = 0.1
# 训练结束后，存储模型参数
save_dir_curr = "checkpoint/{}".format(MODEL_NAME.replace('/','-'))







def read_jsonfile(file_name):
    data = []
    with open(file_name, encoding='UTF8') as f:
        for i in f.readlines():
            data.append(json.loads(i))
    return data

train = pd.DataFrame(read_jsonfile("./data/train.json"))
test = pd.DataFrame(read_jsonfile("./data/testA.json"))

print("train size: {} \ntest size {}".format(len(train),len(test)))

train['text'] = [row['title'] + '，' + row['assignee'] + '，' + row['abstract'] for idx,row in train.iterrows()]
test['text'] = [row['title'] + '，' + row['assignee'] + '，' + row['abstract'] for idx,row in test.iterrows()]
train['concat_len'] = [len(row) for row in train['text']]
test['concat_len'] = [len(row) for row in test['text']]



# 拼接后的文本长度分析
for rate in [0.5,0.75,0.9,0.95,0.99]:
    print("训练数据中{:.0f}%的文本长度小于等于 {:.2f}".format(rate*100,train['concat_len'].quantile(rate)))
plt.title("text length")
sns.distplot(train['concat_len'],bins=10,color='r')
sns.distplot(test['concat_len'],bins=10,color='g')
plt.show()

train_label = train["label_id"].unique()
# 查看标签label分布
plt.figure(figsize=(16,8))
plt.title("label distribution")
sns.countplot(y='label_id',data=train)



# 划分数据集

train_data,valid_data = train_test_split(train,test_size=0.1667,random_state=5)

print("train size: {} \nvalid size {}".format(len(train_data),len(valid_data)))
print("train label: ",sorted(train_data["label_id"].unique()))
print("valid label: ",sorted(valid_data["label_id"].unique()))


# 创建数据迭代器iter
def read(df,istrain=True):
    if istrain:
        for idx,data in df.iterrows():
            yield {
                "words":data['text'],
                "labels":data['label_id']
                }
    else:
        for idx,data in df.iterrows():
            yield {
                "words":data['text'],
                }

# 将生成器传入load_dataset
train_ds = load_dataset(read, df=train_data, lazy=False)
valid_ds = load_dataset(read, df=valid_data, lazy=False)

# 查看数据
for idx in range(1,3):
    print(train_ds[idx])
    print("==="*30)


# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def convert_example(example, tokenizer, max_seq_len=512, mode='train'):
    # 调用tokenizer的数据处理方法把文本转为id
    tokenized_input = tokenizer(example['words'], is_split_into_words=True, max_seq_len=max_seq_len)
    if mode == "test":
        return tokenized_input
    # 把分类标签转为数字id
    tokenized_input['labels'] = [example['labels']]
    return tokenized_input  # 字典形式，包含input_ids、token_type_ids、labels


train_trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    mode='train',
    max_seq_len=max_seq_length)

valid_trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    mode='dev',
    max_seq_len=max_seq_length)

# 映射编码
train_ds.map(train_trans_func, lazy=False)
valid_ds.map(valid_trans_func, lazy=False)

# 初始化BatchSampler
train_batch_sampler = paddle.io.BatchSampler(train_ds, batch_size=train_batch_size, shuffle=True)
valid_batch_sampler = paddle.io.BatchSampler(valid_ds, batch_size=valid_batch_size, shuffle=False)

# 定义batchify_fn
batchify_fn = lambda samples, fn=Dict({
    "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    "labels": Stack(dtype="int32"),
}): fn(samples)

# 初始化DataLoader
train_data_loader = paddle.io.DataLoader(
    dataset=train_ds,
    batch_sampler=train_batch_sampler,
    collate_fn=batchify_fn,
    return_list=True)

valid_data_loader = paddle.io.DataLoader(
    dataset=valid_ds,
    batch_sampler=valid_batch_sampler,
    collate_fn=batchify_fn,
    return_list=True)

# 相同方式构造测试集
test_ds = load_dataset(read, df=test, istrain=False, lazy=False)

test_trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    mode='test',
    max_seq_len=max_seq_length)

test_ds.map(test_trans_func, lazy=False)

test_batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=test_batch_size, shuffle=False)

test_batchify_fn = lambda samples, fn=Dict({
    "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
}): fn(samples)

test_data_loader = paddle.io.DataLoader(
    dataset=test_ds,
    batch_sampler=test_batch_sampler,
    collate_fn=test_batchify_fn,
    return_list=True)





class myModel(ErniePretrainedModel):
    def __init__(self, ernie, num_classes=2, dropout=None):
        super(myModel,self).__init__()
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],num_classes)
        self.apply(self.init_weights)

    def forward(self,
            input_ids,
            token_type_ids=None,
            position_ids=None,
            attention_mask=None):

        _, pooled_output = self.ernie(input_ids,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# 创建model
label_classes = train['label_id'].unique()
model = myModel.from_pretrained(MODEL_NAME,num_classes=len(label_classes))




# 训练总步数
num_training_steps = len(train_data_loader) * epochs

# 学习率衰减策略
lr_scheduler = paddlenlp.transformers.LinearDecayWithWarmup(learning_rate, num_training_steps,warmup_proportion)

decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]


# 定义优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params,
    grad_clip=paddle.nn.ClipGradByGlobalNorm(max_grad_norm))



@paddle.no_grad()
def evaluation(model, data_loader):
    model.eval()
    real_s = []
    pred_s = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        pred_s.extend(probs.argmax(axis=1).numpy())
        real_s.extend(labels.reshape([-1]).numpy())
    score = f1_score(y_pred=pred_s, y_true=real_s, average="macro")
    return score


# 训练阶段
def do_train(model, data_loader):
    print("train ...")
    total_loss = 0.
    model_total_epochs = 0
    best_score = 0.
    num_early_stopping = 0
    if rdrop_coef > 0:
        rdrop_loss = paddlenlp.losses.RDropLoss()
    # 训练
    train_time = time.time()
    valid_time = time.time()
    model.train()
    for epoch in range(0, epochs):

        for step, batch in enumerate(data_loader, start=1):
            input_ids, token_type_ids, labels = batch
            labels = paddle.to_tensor(labels, dtype='int64')
            logits = model(input_ids, token_type_ids)
            # 使用R-drop
            if rdrop_coef > 0:
                logits_2 = model(input_ids=input_ids, token_type_ids=token_type_ids)
                ce_loss = (F.softmax_with_cross_entropy(logits, labels).mean() + F.softmax_with_cross_entropy(logits,
                                                                                                              labels).mean()) * 0.5
                kl_loss = rdrop_loss(logits, logits_2)
                loss = ce_loss + kl_loss * rdrop_coef
            else:
                loss = F.softmax_with_cross_entropy(logits, labels).mean()

            loss.backward()

            total_loss += loss.numpy()

            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            model_total_epochs += 1


        eval_score = evaluation(model, valid_data_loader)
        print("【%.2f%%】validation speed %.2f s" % (
        model_total_epochs / num_training_steps * 100, time.time() - valid_time))
        valid_time = time.time()
        if best_score < eval_score:
            num_early_stopping = 0
            print("eval f1: %.5f f1 update %.5f ---> %.5f " % (eval_score, best_score, eval_score))
            best_score = eval_score
            # 只在score高于0.45的时候保存模型
            if best_score > 0.45:
                # 保存模型
                os.makedirs(save_dir_curr, exist_ok=True)
                save_param_path = os.path.join(save_dir_curr, 'model_best.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                # 保存tokenizer
                tokenizer.save_pretrained(save_dir_curr)
        else:
            num_early_stopping = num_early_stopping + 1
            print("eval f1: %.5f but best f1 %.5f early_stoping_num %d" % (eval_score, best_score, num_early_stopping))
        model.train()
        if num_early_stopping >= early_stopping:
            break
    return best_score




best_score = do_train(model,train_data_loader)



print("best f1 score: %.5f" % best_score)



# logging part
logging_dir = os.path.join('work','submit' )
os.makedirs(logging_dir,exist_ok=True)
logging_name = os.path.join(logging_dir,'run_logging.csv')
os.makedirs(logging_dir,exist_ok=True)

var = [MODEL_NAME, seed, learning_rate, max_seq_length, rdrop_coef, best_score, save_dir_curr]
names = ['model', 'seed', 'lr', "max_len" ,  'rdrop_coef','best_score','save_mode_name']
vars_dict = {k: v for k, v in zip(names, var)}
results = dict(**vars_dict)
keys = list(results.keys())
values = list(results.values())

if not os.path.exists(logging_name):
    ori = []
    ori.append(values)
    logging_df = pd.DataFrame(ori, columns=keys)
    logging_df.to_csv(logging_name, index=False)
else:
    logging_df= pd.read_csv(logging_name)
    new = pd.DataFrame(results, index=[1])
    logging_df = logging_df.append(new, ignore_index=True)
    logging_df.to_csv(logging_name, index=False)

logging_df.tail(10)




# 预测阶段
def do_sample_predict(model,data_loader,is_prob=False):
    model.eval()
    preds = []
    for batch in data_loader:
        input_ids, token_type_ids= batch
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits,axis=1)
        preds.extend(probs.argmax(axis=1).numpy())
    if is_prob:
        return probs
    return preds

# 读取最佳模型
state_dict = paddle.load(os.path.join(save_dir_curr,'model_best.pdparams'))
model.load_dict(state_dict)

# 预测
print("predict start ...")
pred_score = do_sample_predict(model,test_data_loader)
print("predict end ...")




# 例如sumbit_emtion1.csv 就代表日志index为1的提交结果文件
sumbit = pd.DataFrame({"id":test["id"]})
sumbit["label"] = pred_score
file_name = os.path.join(logging_dir, "sumbit_fewshot_{}.csv".format(save_dir_curr.split("/")[1]))
sumbit.to_csv(file_name,index=False)
print("生成提交文件{}".format(file_name))



