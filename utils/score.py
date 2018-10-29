import json
import os
import time


def score(input_file, test_file, result_path):
    pred_file = open(input_file, 'r', encoding='utf-8')
    pred = []
    for line in pred_file:
        if line == '': break
        pair = line.split('][')
        words = eval(pair[0] + ']')
        chars = ''.join(words)
        tags = eval('[' + pair[1])
        rs = []
        start, end = 0, 0
        while start < len(tags) - 1:
            if tags[start].split('_')[0] != 'O':
                while end < len(tags) - 1:
                    if tags[end].split('_')[0] != 'O':
                        end += 1
                    else:
                        break
                if end - start > 1:
                    rs.append(chars[start:end])
                    start = end
            start += 1
            end = start
        rs = list(set(rs))
        pred.append(rs)
    true_file = open(test_file, 'r', encoding='utf-8').readlines()
    result_file = os.path.join(result_path, 'result.txt')
    result_file_f = open(result_file, 'w', encoding='utf-8')
    for i, line in enumerate(true_file):
        info = json.loads(line)
        rs = {'content': info['content'], 'true_label': info['label'], 'pred_label': pred[i]}
        result_file_f.write(json.dumps(rs, ensure_ascii=False) + '\n')
    result_file_f.close()
    true_positive = 0
    positive = 0  # TP + TN
    total_num = 0  # TP + FN
    false_positive = 0
    wrong_file = open(os.path.join(result_path, 'wrong.json'), 'w', encoding='utf-8')
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            rs_line = json.loads(line.strip())
            predict_label = rs_line['pred_label']
            true_label = rs_line['true_label']
            content = rs_line['content']
            if true_label in predict_label:
                true_positive += 1
            else:
                false_positive += 1
                wrong_file.write('{}\t{}\t{}\n'.format(predict_label, true_label, content))
            positive += len(predict_label)
            total_num += 1
    precision = 100.0 * true_positive / positive
    recall = 100.0 * true_positive / total_num
    F1 = 2 * precision * recall / (precision + recall)
    print('Results: right:%d wrong:%d model find:%d total:%d' % (true_positive, false_positive, positive, total_num))
    print('Metrics: Precision:%.3f Recall:%.3f F1:%.3f' % (precision, recall, F1))
    print(time.asctime(time.localtime(time.time())))
