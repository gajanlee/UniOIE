import json
import re
from main import DepTree
from pathlib import Path

class Triplet:

    def __init__(self, subject, predicate, object):
        self.subject, self.predicate, self.object = subject, predicate, object

    def __str__(self):
        return f'{self.subject} {self.predicate} {self.object}'


def generate_modified_triplets(phrase: list):
    if len(phrase) <= 1:
        return phrase[0]['text']

    basic_mod = Triplet(phrase[0]['text'], '@mod', phrase[1]['text'])

    for word in phrase[2:]:
        basic_mod = Triplet(basic_mod, '@mod', word['text'])

    return basic_mod


def _resolve_content(content: dict):
    content = content['content']
    if type(content) is list:
        # 生成mod
        return generate_modified_triplets(content)
    
    return _resolve(content)


def _resolve(element: dict):
    if element == '':
        return ''

    return Triplet(
        _resolve_content(element['subject']),
        _resolve_content(element['predicate']),
        _resolve_content(element['object']),
    )


def resolve_element(ele):
    if type(ele) is str:
        return ele 

    return tuple_to_triplet(ele)


def tuple_to_triplet(tuple):
    subject, predicate, object = tuple

    return Triplet(
        resolve_element(subject),
        resolve_element(predicate),
        resolve_element(object),
    )


# 输入标准树，生成所有潜在triplets，然后进行对比
def load_ann(annotation_file):
    annotation = json.loads(annotation_file.read_text())

    sentence = annotation['sentence']
    relation = _resolve(annotation['relations'][0])
    for r in annotation['relations'][1:]:
        relation = Triplet(relation, '@conj', _resolve(r))

    return sentence, relation


def generate_all_triplets(annotation: Triplet):
    # def _element_to_str(ele):
    #     if type(ele) is str:
    #         return ele
    #     return re.sub(
    #         '\s+', ' ',
    #         f'{_element_to_str(ele)}'
    #     )

    all_triplets = [(
        str(annotation.subject), str(annotation.predicate), str(annotation.object)
    )]

    if type(annotation.subject) is Triplet:
        all_triplets += generate_all_triplets(annotation.subject)
    if type(annotation.predicate) is Triplet:
        all_triplets += generate_all_triplets(annotation.predicate)
    if type(annotation.object) is Triplet:
        all_triplets += generate_all_triplets(annotation.object)

    return all_triplets


def triplets_to_texts(triplets, remove=['@']):
    triplet_texts = []
    for triplet in triplets:
        # if triplet[0] in ['@mod', '@conj']:
            # continue

        text = ' '.join(triplet)
        text = ' '.join([word for word in text.split(' ') if not word in ['@mod', '@conj', '@be', '@null', '@cons', 'BE', 'AND', 'OR', 'CONSTRAIN']])
        text = re.sub(
            '\s+', ' ',
            text,
        ).strip().lower()
        triplet_texts.append(text)

    return set(triplet_texts)

def evaluate(gold_relations: set, predicted_relations: set):
    tp = len(gold_relations & predicted_relations)

    if tp == 0:
        return 0, 0, 0

    recall = tp / len(gold_relations)
    precision = tp / len(predicted_relations)
    f1 = (2 * recall * precision) / (recall + precision)

    return recall, precision, f1


def mean(lst):
    if len(lst) == 0:return 0
    return sum(lst) / len(lst)

def evaluate_token(gold_relations: set, predicted_relations: set):
    recalls = []; precisions = []
    for predict in predicted_relations:
        max_tp = -1; gold_len = 0
        p_words = predict.split(' ')
        for gold in gold_relations:
            g_words = gold.split(' ')

            tp = len(set(p_words) & set(g_words))
            if tp > max_tp:
                max_tp = tp
                gold_len = len(set(g_words))

        recalls.append(tp / gold_len)
        precisions.append(tp / len(p_words))
    
    recall, precision = mean(recalls), mean(precisions)
    f1 = (2 * recall * precision) / (recall + precision) if recall + precision != 0 else 0

    return recall, precision, f1



def relation_to_triplet(relation) -> Triplet:
    subj, pred, obj = relation

    return Triplet(
        subj if type(subj) is str else relation_to_triplet(subj),
        pred if type(pred) is str else relation_to_triplet(pred),
        obj if type(obj) is str else relation_to_triplet(obj),
    )


def evaluate_dataset(annotations):
    for model_name, nest in [
        ('reverb', False),
        ('stanford', False),
        ('clausie', False),
        ('minie', False),
        ('graphene', False),
        ('deepseek-chat', True),
        ('uniOIE', True),
    ]:
        all_relations = json.loads((Path(__file__).parent / f'outputs/{model_name}.output').read_text())
        
        recalls, precisions, f1s = [], [], []
        t_recalls, t_precisions, t_f1s = [], [], []

        for index, (sentence, gold) in annotations.items():
            gold_relations = triplets_to_texts(generate_all_triplets(gold))

            if nest:
            # 嵌套，只有一个根关系
                if all_relations[str(index)] == []:
                    predicted_relations = set()
                else:
                    predicted_relations = triplets_to_texts(generate_all_triplets(tuple_to_triplet(all_relations[str(index)])))

            else:
            # 多个关系
                predicted_relations = set()
                for r in all_relations[str(index)]:
                    predicted_relations |= triplets_to_texts(generate_all_triplets(tuple_to_triplet(r)))

            recall, precision, f1 = evaluate(gold_relations, predicted_relations)
            recalls.append(recall); precisions.append(precision); f1s.append(f1)
            t_recall, t_precision, t_f1 = evaluate_token(gold_relations, predicted_relations)
            t_recalls.append(t_recall); t_precisions.append(t_precision); t_f1s.append(t_f1)

        print(f'=============\n{model_name}\n')
        print('precision: ', sum(precisions) / len(precisions) * 100)
        print('recall: ', sum(recalls) / len(recalls) * 100)
        print('f1: ', sum(f1s) / len(f1s) * 100)

        print('t_precision: ', sum(t_precisions) / len(t_precisions) * 100)
        print('t_recall: ', sum(t_recalls) / len(t_recalls) * 100)
        print('t_f1: ', sum(t_f1s) / len(t_f1s) * 100)




def main():
    annotations = {}

    for index in range(1, 303):
        if index == 50: continue
        file = Path(__file__).parent / 'dev' / f'{index}.json'
        annotations[index] = load_ann(file)

    # dev(302) + hotpotqa + squad + strategyqa
    for index in range(1, 101):
        file = Path(__file__).parent / 'questions/hotpotqa' / f'{index}.json'
        annotations[index + 302] = load_ann(file)
    for index in range(1, 101):
        file = Path(__file__).parent / 'questions/squad' / f'{index}.json'
        annotations[index + 402] = load_ann(file)
    for index in range(1, 101):
        if index == 93: continue
        file = Path(__file__).parent / 'questions/strategyqa' / f'{index}.json'
        annotations[index + 502] = load_ann(file)

    assert len(annotations) == 600

    print('sentence============\n')

    evaluate_dataset({k: v for k, v in annotations.items() if k <= 302})

    print('question============\n')
    evaluate_dataset({k: v for k, v in annotations.items() if k > 302})


if __name__ == '__main__':
    main()


