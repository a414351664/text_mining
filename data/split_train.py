from sklearn.model_selection import train_test_split
import json

def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def save_file(y, x, file_name):
    example = []
    for i in range(len(x)):
        example.append(y[i] + '\t' + x[i])
    # example =
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(example, file, ensure_ascii=False)

# 首先shullf一下，然后再分，结果存起来
contents, labels = [], []
sens = []
with open_file('./cnews/SentiChnDouban.txt') as f:
    for line in f:
        try:
            label, content = line.strip().split('\t')
            if content:
                content = content.replace(' ', '')
                sens.append(content)
                # contents.append(list(content))  # 转为list，一个一个的字了
                labels.append(label)
        except:
            pass
x_train, x_test, y_train, y_test = train_test_split(sens, labels, test_size=0.2, random_state=0)
save_file(y_train, x_train, 'train.json')
save_file(y_test, x_test, 'test.json')
