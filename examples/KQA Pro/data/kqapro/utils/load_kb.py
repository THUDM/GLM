import json
from collections import Counter
from utils.misc import init_vocab
from datetime import date
from queue import Queue
from data.kqapro.utils.value_class import ValueClass

"""
knowledge json format:
    'concepts':
    {
        'id':
        {
            'name': '',
            'instanceOf': ['<concept_id>'],
        }
    },
    'entities': # exclude concepts
    {
        'id': 
        {
            'name': '<entity_name>',
            'instanceOf': ['<concept_id>'],
            'attributes':
            [
                {
                    'key': '<key>',
                    'value': 
                    {
                        'type': 'string'/'quantity'/'date'/'year'
                        'value':  # float or int for quantity, int for year, 'yyyy/mm/dd' for date
                        'unit':   # for quantity
                    },
                    'qualifiers':
                    {
                        '<qk>': 
                        [
                            <qv>, # each qv is a dictionary like value, including keys type,value,unit
                        ]
                    }
                }
            ]
            'relations':
            [
                {
                    'predicate': '<predicate>',
                    'object': '<object_id>', # NOTE: it may be a concept id
                    'direction': 'forward' or 'backward',
                    'qualifiers':
                    {
                        '<qk>': 
                        [
                            <qv>, # each qv is a dictionary like value
                        ]
                    }
                }
            ]
        }
    }
"""
def get_kb_vocab(kb_json, min_cnt=1):
    counter = Counter()
    kb = json.load(open(kb_json))
    for i in kb['concepts']:
        counter.update([i, kb['concepts'][i]['name']])
    for i in kb['entities']:
        counter.update([i, kb['entities'][i]['name']])
        for attr_dict in kb['entities'][i]['attributes']:
            counter.update([attr_dict['key']])
            values = [attr_dict['value']]
            for qk, qvs in attr_dict['qualifiers'].items():
                counter.update([qk])
                values += qvs
            for value in values:
                u = value.get('unit', '')
                if u:
                    counter.update([u])
                counter.update([str(value['value'])])
        for rel_dict in kb['entities'][i]['relations']:
            counter.update([rel_dict['predicate'], rel_dict['direction']])
            values = []
            for qk, qvs in rel_dict['qualifiers'].items():
                counter.update([qk])
                values += qvs
            for value in values:
                u = value.get('unit', '')
                if u:
                    counter.update([u])
                counter.update([str(value['value'])])

    vocab = init_vocab()
    for v, c in counter.items():
        if v and c >= min_cnt and v not in vocab:
            vocab[v] = len(vocab)
    return kb, vocab


def load_as_graph(kb_json, max_desc=200, min_cnt=1):
    kb, vocab = get_kb_vocab(kb_json, min_cnt)
    id2idx = {}
    pred2idx = {}
    node_descs = []
    triples = []
    for i, info in kb['concepts'].items():
        id2idx[i] = len(id2idx)
        desc = [info['name']]
        node_descs.append(desc)
    for i, info in kb['entities'].items():
        id2idx[i] = len(id2idx)
        desc = [info['name']]
        for attr_info in info['attributes']:
            desc.append(attr_info['key'])
            desc.append(str(attr_info['value']['value']))
            u = attr_info['value'].get('unit', '')
            if u:
                desc.append(u)
        node_descs.append(desc)
        for rel_info in info['relations']:
            obj_id = rel_info['object']
            if obj_id not in id2idx:
                continue
            pred = rel_info['predicate']
            if pred not in pred2idx:
                pred2idx[pred] = len(pred2idx)
            pred_idx = pred2idx[pred]
            sub_idx = id2idx[i]
            obj_idx = id2idx[obj_id]
            if rel_info['direction'] == 'forward':
                triples.append((sub_idx, pred_idx, obj_idx))
            else:
                triples.append((obj_idx, pred_idx, sub_idx))
    # encode and pad desc
    for i, desc in enumerate(node_descs):
        desc = [vocab.get(w, vocab['<UNK>']) for w in desc]
        while len(desc) < max_desc:
            desc.append(vocab['<PAD>'])
        node_descs[i] = desc[:max_desc]

    return vocab, node_descs, triples, id2idx, pred2idx



def load_as_key_value(kb_json, min_cnt=1):
    """
    For KVMemNN
    Load each triple (s, r, o) as kv pairs (s+r, o) and (o+r_, s)
    """
    keys = ['<PAD>'] # use <PAD> as the first key
    values = ['<PAD>']
    def add_sro(s, r, o):
        keys.append('{} {}'.format(s, r))
        values.append(o)
        keys.append('{} {}_'.format(o, r))
        values.append(s)

    kb = json.load(open(kb_json))
    for i in kb['concepts']:
        for j in kb['concepts'][i]['instanceOf']:
            s = kb['concepts'][i]['name']
            o = kb['concepts'][j]['name']
            add_sro(s, 'instanceOf', o)
    for i in kb['entities']:
        for j in kb['entities'][i]['instanceOf']:
            s = kb['entities'][i]['name']
            o = kb['concepts'][j]['name']
            add_sro(s, 'instanceOf', o)
        name = kb['entities'][i]['name']
        for attr_dict in kb['entities'][i]['attributes']:
            o = '{} {}'.format(attr_dict['value']['value'], attr_dict['value'].get('unit', ''))
            add_sro(name, attr_dict['key'], o)
            s = '{} {} {}'.format(name, attr_dict['key'], o)
            for qk, qvs in attr_dict['qualifiers'].items():
                for qv in qvs:
                    o = '{} {}'.format(qv['value'], qv.get('unit', ''))
                    add_sro(s, qk, o)

        for rel_dict in kb['entities'][i]['relations']:
            if rel_dict['direction'] == 'backward': # we add reverse relation in add_sro
                continue
            o = kb['entities'].get(rel_dict['object'], kb['concepts'].get(rel_dict['object'], None))
            if o is None: # wtf, why are some objects not in kb?
                continue
            o = o['name']
            add_sro(name, rel_dict['predicate'], o)
            s = '{} {} {}'.format(name, rel_dict['predicate'], o)
            for qk, qvs in rel_dict['qualifiers'].items():
                for qv in qvs:
                    o = '{} {}'.format(qv['value'], qv.get('unit', ''))
                    add_sro(s, qk, o)
    print('length of kv pairs: {}'.format(len(keys)))
    counter = Counter()
    for i in range(len(keys)):
        keys[i] = keys[i].lower().split()
        values[i] = values[i].lower().split()
        counter.update(keys[i])
        counter.update(values[i])

    vocab = init_vocab()
    for v, c in counter.items():
        if v and c >= min_cnt and v not in vocab:
            vocab[v] = len(vocab)
    return vocab, keys, values


class DataForSPARQL(object):
    def __init__(self, kb_path):
        kb = json.load(open(kb_path))
        self.concepts = kb['concepts']
        self.entities = kb['entities']

        # replace adjacent space and tab in name, which may cause errors when building sparql query
        for con_id, con_info in self.concepts.items():
            con_info['name'] = ' '.join(con_info['name'].split())
        for ent_id, ent_info in self.entities.items():
            ent_info['name'] = ' '.join(ent_info['name'].split())

        # get all attribute keys and predicates
        self.attribute_keys = set()
        self.predicates = set()
        self.key_type = {}
        for ent_id, ent_info in self.entities.items():
            for attr_info in ent_info['attributes']:
                self.attribute_keys.add(attr_info['key'])
                self.key_type[attr_info['key']] = attr_info['value']['type']
                for qk in attr_info['qualifiers']:
                    self.attribute_keys.add(qk)
                    for qv in attr_info['qualifiers'][qk]:
                        self.key_type[qk] = qv['type']
        for ent_id, ent_info in self.entities.items():
            for rel_info in ent_info['relations']:
                self.predicates.add(rel_info['predicate'])
                for qk in rel_info['qualifiers']:
                    self.attribute_keys.add(qk)
                    for qv in rel_info['qualifiers'][qk]:
                        self.key_type[qk] = qv['type']
        self.attribute_keys = list(self.attribute_keys)
        self.predicates = list(self.predicates)
        # Note: key_type is one of string/quantity/date, but date means the key may have values of type year
        self.key_type = { k:v if v!='year' else 'date' for k,v in self.key_type.items() }

        # parse values into ValueClass object
        for ent_id, ent_info in self.entities.items():
            for attr_info in ent_info['attributes']:
                attr_info['value'] = self._parse_value(attr_info['value'])
                for qk, qvs in attr_info['qualifiers'].items():
                    attr_info['qualifiers'][qk] = [self._parse_value(qv) for qv in qvs]
        for ent_id, ent_info in self.entities.items():
            for rel_info in ent_info['relations']:
                for qk, qvs in rel_info['qualifiers'].items():
                    rel_info['qualifiers'][qk] = [self._parse_value(qv) for qv in qvs]

    def _parse_value(self, value):
        if value['type'] == 'date':
            x = value['value']
            p1, p2 = x.find('/'), x.rfind('/')
            y, m, d = int(x[:p1]), int(x[p1+1:p2]), int(x[p2+1:])
            result = ValueClass('date', date(y, m, d))
        elif value['type'] == 'year':
            result = ValueClass('year', value['value'])
        elif value['type'] == 'string':
            result = ValueClass('string', value['value'])
        elif value['type'] == 'quantity':
            result = ValueClass('quantity', value['value'], value['unit'])
        else:
            raise Exception('unsupport value type')
        return result

    def get_direct_concepts(self, ent_id):
        """
        return the direct concept id of given entity/concept
        """
        if ent_id in self.entities:
            return self.entities[ent_id]['instanceOf']
        elif ent_id in self.concepts:
            return self.concepts[ent_id]['instanceOf']
        else:
            raise Exception('unknown id')

    def get_all_concepts(self, ent_id):
        """
        return a concept id list
        """
        ancestors = []
        q = Queue()
        for c in self.get_direct_concepts(ent_id):
            q.put(c)
        while not q.empty():
            con_id = q.get()
            ancestors.append(con_id)
            for c in self.concepts[con_id]['instanceOf']:
                q.put(c)

        return ancestors

    def get_name(self, ent_id):
        if ent_id in self.entities:
            return self.entities[ent_id]['name']
        elif ent_id in self.concepts:
            return self.concepts[ent_id]['name']
        else:
            return None

    def is_concept(self, ent_id):
        return ent_id in self.concepts

    def get_attribute_facts(self, ent_id, key=None, unit=None):
        if key:
            facts = []
            for attr_info in self.entities[ent_id]['attributes']:
                if attr_info['key'] == key:
                    if unit:
                        if attr_info['value'].unit == unit:
                            facts.append(attr_info)
                    else:
                        facts.append(attr_info)
        else:
            facts = self.entities[ent_id]['attributes']
        facts = [(f['key'], f['value'], f['qualifiers']) for f in facts]
        return facts

    def get_relation_facts(self, ent_id):
        facts = self.entities[ent_id]['relations']
        facts = [(f['predicate'], f['object'], f['direction'], f['qualifiers']) for f in facts]
        return facts
