from google.colab import drive
import os
import sys

# Mount Google Drive
drive.mount('/content/gdrive/')

# List the contents of the specified directory
print(os.listdir('/content/gdrive/My Drive/CSE-143-A3'))

# Add the directory to the system path
sys.path.append('/content/gdrive/My Drive/CSE-143-A3')

# Change to the working directory
%cd '/content/gdrive/My Drive/CSE-143-A3'

import random
from conlleval import evaluate as conllevaluate

directory = '/content/gdrive/My Drive/CSE-143-A3'

class FeatureVector(object):
    def __init__(self, fdict):
        self.fdict = fdict

    def times_plus_equal(self, scalar, v2):
        for key, value in v2.fdict.items():
            self.fdict[key] = scalar * value + self.fdict.get(key, 0)

    def dot_product(self, v2):
        retval = 0
        for key, value in v2.fdict.items():
            retval += value * self.fdict.get(key, 0)
        return retval

    def write_to_file(self, filename):
        print('Writing to ' + filename)
        with open(filename, 'w', encoding='utf-8') as f:
            for key, value in self.fdict.items():
                f.write('{} {}\n'.format(key, value))

    def read_from_file(self, filename):
        self.fdict = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                txt = line.split()
                self.fdict[txt[0]] = float(txt[1])

class Features(object):
    def __init__(self, inputs, feature_names):
        self.feature_names = feature_names
        self.inputs = inputs

    def compute_features(self, cur_tag, pre_tag, i):
        feats = FeatureVector({})
        if 'tag' in self.feature_names:
            feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag: 1}))
        if 'prev_tag' in self.feature_names:
            feats.times_plus_equal(1, FeatureVector({'ti=' + cur_tag + "+ti-1=" + pre_tag: 1}))
        if 'current_word' in self.feature_names:
            feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag + '+w=' + self.inputs['tokens'][i]: 1}))
        return feats

def decode(input_length, tagset, score):
    dp = [{} for _ in range(input_length)]
    backtrace = [{} for _ in range(input_length)]

    dp[0]['<START>'] = 0

    for i in range(1, input_length):
        for curr_tag in (tagset if i < input_length - 1 else ['<STOP>']):
            max_score = float('-inf')
            best_prev_tag = None
            for prev_tag in (['<START>'] if i == 1 else tagset):
                current_score = dp[i - 1][prev_tag] + score(curr_tag, prev_tag, i)
                if current_score > max_score:
                    max_score = current_score
                    best_prev_tag = prev_tag
            dp[i][curr_tag] = max_score
            backtrace[i][curr_tag] = best_prev_tag

    best_path = ['<STOP>']
    for i in range(input_length - 1, 0, -1):
        best_path.append(backtrace[i][best_path[-1]])

    best_path.reverse()
    return best_path

def compute_score(tag_seq, input_length, score):
    total_score = 0
    for i in range(1, input_length):
        total_score += score(tag_seq[i], tag_seq[i - 1], i)
    return total_score

def compute_features(tag_seq, input_length, features):
    feats = FeatureVector({})
    for i in range(1, input_length):
        feats.times_plus_equal(1, features.compute_features(tag_seq[i], tag_seq[i - 1], i))
    return feats

def sgd(training_size, epochs, gradient, parameters, training_observer):
    i = 0
    while i < epochs:
        print('i=' + str(i))
        data_indices = [i for i in range(training_size)]
        random.shuffle(data_indices)
        counter = 0
        for t in data_indices:
            if counter % 1000 == 0:
                print('Item ' + str(counter))
            parameters.times_plus_equal(-1, gradient(t))
            counter += 1
        i += 1
        training_observer(i, parameters)
    return parameters

def train(data, feature_names, tagset, epochs):
    parameters = FeatureVector({})

    def perceptron_gradient(i):
        inputs = data[i]
        input_len = len(inputs['tokens'])
        gold_labels = inputs['gold_tags']
        features = Features(inputs, feature_names)

        def score(cur_tag, pre_tag, i):
            return parameters.dot_product(features.compute_features(cur_tag, pre_tag, i))

        tags = decode(input_len, tagset, score)
        fvector = compute_features(tags, input_len, features)
        fvector.times_plus_equal(-1, compute_features(gold_labels, input_len, features))
        return fvector

    def training_observer(epoch, parameters):
        dev_data = read_data('ner.dev')
        (_, _, f1) = evaluate(dev_data, parameters, feature_names, tagset)
        write_predictions('ner.dev.out' + str(epoch), dev_data, parameters, feature_names, tagset)
        parameters.write_to_file('model.iter' + str(epoch))
        return f1

    return sgd(len(data), epochs, perceptron_gradient, parameters, training_observer)

def predict(inputs, input_len, parameters, feature_names, tagset):
    features = Features(inputs, feature_names)

    def score(cur_tag, pre_tag, i):
        return parameters.dot_product(features.compute_features(cur_tag, pre_tag, i))

    return decode(input_len, tagset, score)

def make_data_point(sent):
    dic = {}
    sent = [s.strip().split() for s in sent]
    dic['tokens'] = ['<START>'] + [s[0] for s in sent] + ['<STOP>']
    dic['pos'] = ['<START>'] + [s[1] for s in sent] + ['<STOP>']
    dic['NP_chunk'] = ['<START>'] + [s[2] for s in sent] + ['<STOP>']
    dic['gold_tags'] = ['<START>'] + [s[3] for s in sent] + ['<STOP>']
    return dic

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        sent = []
        for line in f.readlines():
            if line.strip():
                sent.append(line)
            else:
                data.append(make_data_point(sent))
                sent = []
        data.append(make_data_point(sent))
    return data

def write_predictions(out_filename, all_inputs, parameters, feature_names, tagset):
    with open(out_filename, 'w', encoding='utf-8') as f:
        for inputs in all_inputs:
            input_len = len(inputs['tokens'])
            tag_seq = predict(inputs, input_len, parameters, feature_names, tagset)
            for i, tag in enumerate(tag_seq[1:-1]):
                f.write(' '.join([inputs['tokens'][i + 1], inputs['pos'][i + 1], inputs['NP_chunk'][i + 1], inputs['gold_tags'][i + 1], tag]) + '\n')
            f.write('\n')

def evaluate(data, parameters, feature_names, tagset):
    all_gold_tags = []
    all_predicted_tags = []
    for inputs in data:
        all_gold_tags.extend(inputs['gold_tags'][1:-1])
        input_len = len(inputs['tokens'])
        all_predicted_tags.extend(predict(inputs, input_len, parameters, feature_names, tagset)[1:-1])
    return conllevaluate(all_gold_tags, all_predicted_tags)

def main_predict(data_filename, model_filename):
    data = read_data(data_filename)
    parameters = FeatureVector({})
    parameters.read_from_file(model_filename)

    tagset = ['B-PER', 'B-LOC', 'B-ORG', 'B-MISC', 'I-PER', 'I-LOC', 'I-ORG', 'I-MISC', 'O']
    feature_names = ['tag', 'prev_tag', 'current_word']

    write_predictions(data_filename + '.out', data, parameters, feature_names, tagset)
    precision, recall, f1 = evaluate(data, parameters, feature_names, tagset)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    return precision, recall, f1

# Path to dev data and model file
dev_data_path = '/content/gdrive/My Drive/CSE-143-A3/ner.dev'
model_path = '/content/gdrive/My Drive/CSE-143-A3/model.simple'

# Run the prediction and evaluation
main_predict(dev_data_path, model_path)
