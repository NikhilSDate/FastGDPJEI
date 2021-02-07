import spacy
from spacy import displacy
import json
import nltk
from queue import Queue
from spacy.symbols import *


def get_entities(doc):
    entities = []
    for token in doc:
        if token.text.isupper() and token.pos_ != DET:
            if token.text[-1:] == '.':
                entities.append((token, token.idx, token.idx + len(token) - 2))
            else:
                entities.append((token, token.idx, token.idx + len(token) - 1))
    return entities


def add_type_annotations():
    with open('../aaai/problem_texts.json') as file:
        json_obj = json.loads(file.read())
        for i in range(10, 20):
            problem = json_obj[str(i)]
            problem_text = problem['text']
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(problem_text)
            entities = get_entities(doc)

            entity_list = list()
            print(problem['text'])
            for entity in entities:
                entity_token, start, end = entity
                type = input(entity_token.text + ": ")
                entity_list.append((start, end, type))
            problem['entities'] = entity_list
    with open('../aaai/problem_texts.json', 'w') as outfile:
        json.dump(json_obj, outfile, indent=4)


def find_path(token1, token2):
    nodes_queue = Queue()
    node_parents = dict()
    path = list()
    nodes_queue.put(token1)
    node_parents[token1] = None
    while not nodes_queue.empty():
        v = nodes_queue.get()
        if v == token2:
            path_token = v
            path.append(path_token)
            while node_parents[path_token] is not None:
                path.append(node_parents[path_token])
                path_token = node_parents[path_token]
        else:
            for child in v.lefts:
                if child not in node_parents.keys():
                    nodes_queue.put(child)
                    node_parents[child] = v
            for child in v.rights:
                if child not in node_parents.keys():
                    nodes_queue.put(child)
                    node_parents[child] = v
            if v.head not in node_parents.keys():
                nodes_queue.put(v.head)
                node_parents[v.head] = v

    return path[::-1]


def get_entity_type_phrases(doc):
    entity_set = set([entity_token for entity_token, _, _ in get_entities(doc)])
    entity_phrases = list()
    type_phrases = list()
    print(entity_set)
    for np in doc.noun_chunks:
        token_set = set([token for token in np])
        print(token_set)
        print(entity_set)
        entities_in_np = entity_set.intersection(token_set)
        print(entities_in_np)
        if len(entities_in_np) == 1:
            entity_phrases.append(np)
            entity = next(iter(entities_in_np))
            print(entity)
            entity_set.remove(entity)
            print(entity_set)
        elif len(entities_in_np) == 0:
            type_phrases.append(np)
    for entity_left in entity_set:
        entity_phrases.append(entity_left)
    return entity_phrases, type_phrases


# with open('aaai/problem_texts.json') as file:
#     json = json.loads(file.read())
#     text = json[str(3)]['text']

# print(get_entity_type_phrases(doc))
text='If the measure of angle ACD is equal to the measure of angle BCE and AB is perpendicular to BC'
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
for np in doc.noun_chunks:
    print(np)
# displacy.serve(doc)
# print(get_entity_type_phrases(doc))
print(find_path(doc[15],doc[19]))