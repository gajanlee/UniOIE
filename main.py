import stanza
from collections import namedtuple
from pathlib import Path
from tqdm import tqdm
import json

Word = namedtuple('Word', ['text', 'pos', 'index'])
Node = namedtuple('Node', ['word', 'children'])

stanza_parser = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', logging_level="ERROR", use_gpu=True, download_method=None)

class DepTree:

    def __init__(self, sentence):
        basic_dependency = stanza_parser(sentence)
        self._construct_word_list(basic_dependency)
        self._construct_dependency_tree(basic_dependency)

        self.relation = self._extract_uniOIE()

        for deprels in self.tree.values():
            for index, deprel, used in deprels:
                if not used:
                    print(deprel)


    def _construct_word_list(self, basic_dependency):
        self.word_list = {'': Word('', '', 0)}
        # print(basic_dependency.to_dict()[0])
        for index, data in enumerate(basic_dependency.to_dict()[0], 1): 
            word, pos_tag = data["text"], data["xpos"]
            if data["upos"] == "X":     # X means other, which can be regarded as a nominal
                pos_tag = 'NN'
            if data["upos"] in ["SYM", "PUNCT"]:
                pos_tag = "SYM"
        
            self.word_list[data['id']] = Word(word, pos_tag, index)

    def _construct_dependency_tree(self, basic_dependency):
        self.tree = {}
        for index, data in enumerate(basic_dependency.to_dict()[0], 1): 
            child, deprel = self.word_list[data["id"]], data["deprel"]
            deprel = deprel.split(':')[0]

            if deprel == 'root':
                self.root = child.index
                continue
            
            head = self.word_list[data['head']]
            # True代表该关系没有被处理过
            self.tree[head.index] = self.tree.get(head.index, []) + [[child.index, deprel, True]]
        
        # print(self.tree)

    def _extract_uniOIE(self):
        relation_id = self._parse_clause(self.root)
        # print(relation_id)

        def _relation_id_to_text(rel):
            return [_relation_id_to_text(ele)
                    if type(ele) is tuple else 
                    (ele if type(ele) is str else self.word_list[ele].text)
                    for ele in rel]

        relation = _relation_id_to_text(relation_id)
        # print(relation)

        return relation

    def _parse_clause(self, node):
        # 并列的clause
        if (
            self._has_dep(node, 'conj') and 
            self._has_dep(self._has_dep(node, 'conj'), 'nsubj')
        ):
                conj_node = self._get_child(node, 'conj')
                rel = self._get_child(conj_node, ['cc'])
                rel = '@conj' if not rel else rel
                print(self.tree)
                return (
                    self._parse_clause(node), 
                    rel,
                    self._parse_clause(conj_node),
                )

        # SVC
        if (cop_node := self._get_child(node, 'cop')):
            return (
                self._parse_phrase(self._get_child(node, 'nsubj')), 
                self._parse_phrase(cop_node),
                self._parse_phrase(node),
            )

        if (comp_node := self._get_child(node, ['xcomp', 'ccomp'])):
            rel = self._get_child(comp_node, ['case', 'mark'])
            rel = '@cons' if not rel else rel

            return (
                self._parse_clause(node),
                rel,
                self._parse_clause(comp_node),
            )

        if (advcl_node := self._get_child(node, 'advcl')):
            return (
                self._parse_clause(node),
                self._get_child(advcl_node, ['case', 'mark']),
                self._parse_clause(advcl_node)
            )

        # SVOO
        if (iobj_node := self._get_child(node, 'iobj')):
            return ((
                    self._parse_phrase(self._get_child(node, 'nsubj')),
                    self._parse_phrase(node),
                    self._parse_phrase(iobj_node)
                ),
                '@cons',
                self._parse_phrase(self._get_child(node, 'obj')),
            )
        # I give an apple to her
        # give -> obl -> her
        if (obl_node := self._get_child(node, 'obl')):
            return ((
                    self._parse_phrase(self._get_child(node, 'nsubj')),
                    self._parse_phrase(node),
                    self._parse_phrase(self._get_child(node, 'obj'))
                ),
                self._get_child(obl_node, ['case', 'mark']),
                self._parse_phrase(obl_node)
            )

        # SVO
        if self._has_dep(node, 'obj'):
            return (
                self._parse_phrase(self._get_child(node, 'nsubj')),
                self._parse_phrase(node),
                self._parse_phrase(self._get_child(node, 'obj')),
            )
        # SV
        else:
            return (
                self._parse_phrase(self._get_child(node, 'nsubj')),
                self._parse_phrase(node),
                '',
            )

    def _parse_phrase(self, node):
        if (conj_node := self._get_child(node, 'conj')):
            rel = self._get_child(conj_node, ['cc'])
            rel = '@conj' if not rel else rel

            return (
                self._parse_phrase(node),
                rel,
                self._parse_phrase(conj_node),
            )

        if (mod_node := self._get_child(node, ['nummod', 'amod', 'det', 'advmod', 'compound', 'aux'])):
            return (
                self._parse_phrase(mod_node),
                '@mod',
                self._parse_phrase(node),
            )

        if (noun_mod_node := self._get_child(node, ['nmod'])):
            rel = self._get_child(noun_mod_node, ['case'])
            rel = '@mod' if not rel else rel

            return (
                self._parse_phrase(noun_mod_node),
                rel,
                self._parse_phrase(node),
            )
    
        # 定语从句
        if (acl_node := self._get_child(node, 'acl')):
            rel = self._get_child(acl_node, 'case')
            rel = '@cons' if not rel else rel

            return (
                self._parse_phrase(node),
                rel,
                self._parse_clause(acl_node)
            )

        if (appos_node := self._get_child(node, 'appos')):
            return (
                self._parse_phrase(node),
                '@be',
                self._parse_phrase(appos_node),
            )

        return node

    def _has_dep(self, node, deprels):
        if type(deprels) is str:
            deprels = [deprels] 

        for child, dep, available in self.tree.get(node, []):
            if dep in deprels and available:
                return child

        return 0

    def _get_child(self, node, deprels):
        if type(deprels) is str:
            deprels = [deprels] 
        # 只能获取一次
        for index, (child, dep, available) in enumerate(self.tree.get(node, [])):
            if available and dep in deprels:
                self.tree[node][index][2] = False
                return child

        return ''


# 输入一个句子，输出一棵树
# 根据依赖关系的算法
def convert_UniOIE(sentence: str):
    # 把句子转换为dependency
    relation = DepTree(sentence).relation

    return relation


def test():
    # SV
    assert (
        convert_UniOIE('I am erected.') == 
        ['I', ['am', '@mod', 'erected'], '']
    )
    # SVO
    assert (
        convert_UniOIE('I love you') == 
        ['I', 'love', 'you']
    )
    # SVC
    assert (
        convert_UniOIE('I am interested.') ==
        ['I', 'am', 'interested']
    )
    # SVOA
    assert (
        convert_UniOIE('I marry her if she is good.') == 
        [['I', 'marry', 'her'], 'if', ['she', 'is', 'good']]
    )
    # SVOO
    # iobj: give->her
    # obj: give->apple
    assert (
        convert_UniOIE('I give her an apple.') ==
        [['I', 'give', 'her'], '@cons', ['an', '@mod', 'apple']]
    )
    # SVOA
    assert (
        convert_UniOIE('I give an animal to her.') ==
        [['I', 'give', ['an', '@mod', 'animal']], 'to', 'her']
    )
    # SVOA
    assert (
        convert_UniOIE('I ate an orange in the morning.') ==
        [['I', 'ate', ['an', '@mod', 'orange']], 'in', ['the', '@mod', 'morning']]
    )
    exit()

    # 并列句
    assert (
        convert_UniOIE('I ate an orange and he ate an apple.') ==
        [['I', 'ate', ['an', '@mod', 'orange']], 'and', ['he', 'ate', ['an', '@mod', 'apple']]]
    )
    # 动词并列
    assert (
        convert_UniOIE('I fuck, resort and ate the orange.') == ['I', [['fuck', 'and', 'ate'], '@conj', 'resort'], '']
    )

    # 名词并列，动词修饰，副词修饰，形容词修饰
    rel = convert_UniOIE('I quickly ate a big apple and a very big orange.')
    assert rel == ['I', ['quickly', '@mod', 'ate'], [['a', '@mod', ['big', '@mod', 'apple']], 'and', ['a', '@mod', [['very', '@mod', 'big'], '@mod', 'orange']]]]


    # 定语从句
    assert (
        convert_UniOIE('I ate an orange that is from Alice.') ==
        ['I', 'ate', ['an', '@mod', ['orange', 'from', ['that', 'is', 'Alice']]]]
    )

    # 补语从句
    # He says that you like to swim
    # 保持最细粒度的语义，缺失主语也是一种主语
    # 句法太复杂
    assert (
        convert_UniOIE('I says that you like to swim') == [['I', 'says', ''], 'that', [['you', 'like', ''], 'to', ['', 'swim', '']]]
    )



if __name__ == '__main__':
    output_file = Path('outputs/uniOIE.output')

    all_relations = {}
    for index, sentence in enumerate(tqdm((Path(__file__).parent / 'all.txt').read_text().split('\n')), 1):
        relation = DepTree(sentence).relation

        all_relations[index] = relation

        output_file.write_text(json.dumps(all_relations, indent=2))
    
