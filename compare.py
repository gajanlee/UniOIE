import requests
import json
from pathlib import Path
from tqdm import tqdm

# Graphene
def graphene(sentences):
    # 定义请求的 URL
    url = "http://localhost:8080/relationExtraction/text"

    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    all_relations = {}
    for index, sentence in tqdm(enumerate(sentences, 1), desc='graphene'):
        # 定义请求体
        data = {
            "text": sentence,
            "doCoreference": "false",
            "isolateSentences": "false",
            "format": "DEFAULT"
        }

        # 发送 POST 请求
        response = requests.post(url, headers=headers, json=data)

        relations = []
        for relation in json.loads(response.text)['sentences'][0]['extractionMap'].values():
            relations.append([
                relation['arg1'],
                relation['relation'],
                relation['arg2'],
            ])

        all_relations[index] = relations

    return all_relations


def stanford(sentences):
    from openie import StanfordOpenIE
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }

    all_relations = {i: [] for i in range(1, 603)}
    with StanfordOpenIE(properties=properties, install_dir_path='../stanfordOIE') as client:
        for index, sentence in tqdm(enumerate(sentences, 1), desc='stanford'):
            relations = []
            for relation in client.annotate(sentence):
                relations.append([
                    relation['subject'],
                    relation['relation'],
                    relation['object'],
                ])

            all_relations[index] = relations

    return all_relations


def clausie():
    pass


def minie(sentences):
    import os
    from pathlib import Path
    os.environ['CLASSPATH'] = '/home/lee/openIE/LinkingConcepts/zoo/minie/miniepy/target/minie-0.0.1-SNAPSHOT.jar'

    from jnius import autoclass

    CoreNLPUtils = autoclass('de.uni_mannheim.utils.coreNLP.CoreNLPUtils')
    AnnotatedProposition = autoclass('de.uni_mannheim.minie.annotation.AnnotatedProposition')
    MinIE = autoclass('de.uni_mannheim.minie.MinIE')
    StanfordCoreNLP = autoclass('edu.stanford.nlp.pipeline.StanfordCoreNLP')
    String = autoclass('java.lang.String')

    parser = CoreNLPUtils.StanfordDepNNParser()

    MinIEMode = autoclass('de.uni_mannheim.minie.MinIE$Mode')
    
    all_relations = {}
    for index, sentence in enumerate(sentences):
        output = MinIE(String(sentence), parser, MinIEMode.SAFE)

        relations = []
        for ap in output.getPropositions().elements():
            if ap is not None:
                subject, indicator, *object = ap.getTripleAsString().replace('"', '').split('\t')
                object = ' '.join([obj for obj in object if obj])
                relations.append([subject, indicator, object])

        all_relations[index] = relations

    return all_relations


def generate_triplets():
    graphene_output_file = Path('outputs/graphene.output')
    stanford_output_file = Path('outputs/stanford.output')
    minie_output_file = Path('outputs/minie.output')

    # all.txt
    # dev(302) + hotpotqa + squad + strategyqa
    sentences = (Path(__file__).parent / 'all.txt').read_text().split('\n')

    graphene_output = graphene(sentences)
    graphene_output_file.write_text(json.dumps(graphene_output, indent=2))

    stanford_output = stanford(sentences)
    stanford_output_file.write_text(json.dumps(stanford_output, indent=2))

    minie_output = stanford(sentences)
    minie_output_file.write_text(json.dumps(minie_output, indent=2))


if __name__ == '__main__':
    generate_triplets()
