import rdflib
from rdflib import URIRef, BNode, Literal, XSD
from rdflib.plugins.stores import sparqlstore
from itertools import chain
from tqdm import tqdm
import argparse

from data.kqapro.utils.load_kb import DataForSPARQL
from data.kqapro.utils.value_class import ValueClass


virtuoso_address = "http://166.111.68.66:25890/sparql"


def legal(s):
    # convert predicate and attribute keys to legal format
    return s.replace(' ', '_')

def esc_escape(s):
    '''
    Why we need this:
    If there is an escape in Literal, such as '\EUR', the query string will be something like '?pv <pred:value> "\\EUR"'.
    However, in virtuoso engine, \\ is connected with E, and \\E forms a bad escape sequence.
    So we must repeat \\, and virtuoso will consider "\\\\EUR" as "\EUR".

    Note this must be applied before esc_quot, as esc_quot will introduce extra escapes.
    '''
    return s.replace('\\', '\\\\')

def esc_quot(s):
    '''
    Why we need this:
    We use "<value>" to represent a literal value in the sparql query.
    If the <value> has a double quotation mark itself, we must escape it to make sure the query is valid for the virtuoso engine.
    '''
    return s.replace('"', '\\"')

class SparqlEngine():
    gs1 = None
    PRED_INSTANCE = 'pred:instance_of'
    PRED_NAME = 'pred:name'

    PRED_VALUE = 'pred:value'       # link packed value node to its literal value
    PRED_UNIT = 'pred:unit'         # link packed value node to its unit

    PRED_YEAR = 'pred:year'         # link packed value node to its year value, which is an integer
    PRED_DATE = 'pred:date'         # link packed value node to its date value, which is a date

    PRED_FACT_H = 'pred:fact_h'     # link qualifier node to its head
    PRED_FACT_R = 'pred:fact_r'
    PRED_FACT_T = 'pred:fact_t'

    SPECIAL_PREDICATES = (PRED_INSTANCE, PRED_NAME, PRED_VALUE, PRED_UNIT, PRED_YEAR, PRED_DATE, PRED_FACT_H, PRED_FACT_R, PRED_FACT_T)
    def __init__(self, data, ttl_file=''):
        self.nodes = nodes = {}
        for i in chain(data.concepts, data.entities):
            nodes[i] = URIRef(i)
        for p in chain(data.predicates, data.attribute_keys, SparqlEngine.SPECIAL_PREDICATES):
            nodes[p] = URIRef(legal(p))
        
        self.graph = graph = rdflib.Graph()

        for i in chain(data.concepts, data.entities):
            name = data.get_name(i)
            graph.add((nodes[i], nodes[SparqlEngine.PRED_NAME], Literal(name)))

        for ent_id in tqdm(data.entities, desc='Establishing rdf graph'):
            for con_id in data.get_all_concepts(ent_id):
                graph.add((nodes[ent_id], nodes[SparqlEngine.PRED_INSTANCE], nodes[con_id]))
            for (k, v, qualifiers) in data.get_attribute_facts(ent_id):
                h, r = nodes[ent_id], nodes[k]
                t = self._get_value_node(v)
                graph.add((h, r, t))
                fact_node = self._new_fact_node(h, r, t)

                for qk, qvs in qualifiers.items():
                    for qv in qvs:
                        h, r = fact_node, nodes[qk]
                        t = self._get_value_node(qv)
                        if len(list(graph[t])) == 0:
                            print(t)
                        graph.add((h, r, t))

            for (pred, obj_id, direction, qualifiers) in data.get_relation_facts(ent_id):
                if direction == 'backward':
                    if data.is_concept(obj_id):
                        h, r, t = nodes[obj_id], nodes[pred], nodes[ent_id]
                    else:
                        continue
                else:
                    h, r, t = nodes[ent_id], nodes[pred], nodes[obj_id]
                graph.add((h, r, t))
                fact_node = self._new_fact_node(h, r, t)
                for qk, qvs in qualifiers.items():
                    for qv in qvs:
                        h, r = fact_node, nodes[qk]
                        t = self._get_value_node(qv)
                        graph.add((h, r, t))

        if ttl_file:
            print('Save graph to {}'.format(ttl_file))
            graph.serialize(ttl_file, format='turtle')


    def _get_value_node(self, v):
        # we use a URIRef node, because we need its reference in query results, which is not supported by BNode
        if v.type == 'string':
            node = BNode()
            self.graph.add((node, self.nodes[SparqlEngine.PRED_VALUE], Literal(v.value)))
            return node
        elif v.type == 'quantity': 
            # we use a node to pack value and unit
            node = BNode()
            self.graph.add((node, self.nodes[SparqlEngine.PRED_VALUE], Literal(v.value, datatype=XSD.double)))
            self.graph.add((node, self.nodes[SparqlEngine.PRED_UNIT], Literal(v.unit)))
            return node
        elif v.type == 'year':
            node = BNode()
            self.graph.add((node, self.nodes[SparqlEngine.PRED_YEAR], Literal(v.value)))
            return node
        elif v.type == 'date':
            # use a node to pack year and date
            node = BNode()
            self.graph.add((node, self.nodes[SparqlEngine.PRED_YEAR], Literal(v.value.year)))
            self.graph.add((node, self.nodes[SparqlEngine.PRED_DATE], Literal(v.value, datatype=XSD.date)))
            return node

    def _new_fact_node(self, h, r, t):
        node = BNode()
        self.graph.add((node, self.nodes[SparqlEngine.PRED_FACT_H], h))
        self.graph.add((node, self.nodes[SparqlEngine.PRED_FACT_R], r))
        self.graph.add((node, self.nodes[SparqlEngine.PRED_FACT_T], t))
        return node


def query_virtuoso(q):
    endpoint = virtuoso_address
    store=sparqlstore.SPARQLUpdateStore(endpoint)
    gs = rdflib.ConjunctiveGraph(store)
    gs.open((endpoint, endpoint))
    # gs1 = gs.get_context(rdflib.URIRef(virtuoso_graph_uri))
    res = gs.query(q)
    return res



def get_sparql_answer(sparql, data):
    """
    data: DataForSPARQL object, we need the key_type
    """
    try:
        # infer the parse_type based on sparql
        if sparql.startswith('SELECT DISTINCT ?e') or sparql.startswith('SELECT ?e'):
            parse_type = 'name'
        elif sparql.startswith('SELECT (COUNT(DISTINCT ?e)'):
            parse_type = 'count'
        elif sparql.startswith('SELECT DISTINCT ?p '):
            parse_type = 'pred'
        elif sparql.startswith('ASK'):
            parse_type = 'bool'
        else:
            tokens = sparql.split()
            tgt = tokens[2]
            for i in range(len(tokens)-1, 1, -1):
                if tokens[i]=='.' and tokens[i-1]==tgt:
                    key = tokens[i-2]
                    break
            key = key[1:-1].replace('_', ' ')
            t = data.key_type[key]
            parse_type = 'attr_{}'.format(t)

        parsed_answer = None
        res = query_virtuoso(sparql)
        if res.vars:
            res = [[binding[v] for v in res.vars] for binding in res.bindings]
            if len(res) != 1:
                return None
        else:
            res = res.askAnswer
            assert parse_type == 'bool'
        
        if parse_type == 'name':
            node = res[0][0]
            sp = 'SELECT DISTINCT ?v WHERE {{ <{}> <{}> ?v .  }}'.format(node, SparqlEngine.PRED_NAME)
            res = query_virtuoso(sp)
            res = [[binding[v] for v in res.vars] for binding in res.bindings]
            name = res[0][0].value
            parsed_answer = name
        elif parse_type == 'count':
            count = res[0][0].value
            parsed_answer = str(count)
        elif parse_type.startswith('attr_'):
            node = res[0][0]
            v_type = parse_type.split('_')[1]
            unit = None
            if v_type == 'string':
                sp = 'SELECT DISTINCT ?v WHERE {{ <{}> <{}> ?v .  }}'.format(node, SparqlEngine.PRED_VALUE)
            elif v_type == 'quantity':
                # Note: For those large number, ?v is truncated by virtuoso (e.g., 14756087 to 1.47561e+07)
                # To obtain the accurate ?v, we need to cast it to str
                sp = 'SELECT DISTINCT ?v,?u,(str(?v) as ?sv) WHERE {{ <{}> <{}> ?v ; <{}> ?u .  }}'.format(node, SparqlEngine.PRED_VALUE, SparqlEngine.PRED_UNIT)
            elif v_type == 'year':
                sp = 'SELECT DISTINCT ?v WHERE {{ <{}> <{}> ?v .  }}'.format(node, SparqlEngine.PRED_YEAR)
            elif v_type == 'date':
                sp = 'SELECT DISTINCT ?v WHERE {{ <{}> <{}> ?v .  }}'.format(node, SparqlEngine.PRED_DATE)
            else:
                raise Exception('unsupported parse type')
            res = query_virtuoso(sp)
            res = [[binding[v] for v in res.vars] for binding in res.bindings]
            # if there is no specific date, then convert the type to year
            if len(res)==0 and v_type == 'date':
                v_type = 'year'
                sp = 'SELECT DISTINCT ?v WHERE {{ <{}> <{}> ?v .  }}'.format(node, SparqlEngine.PRED_YEAR)
                res = query_virtuoso(sp)
                res = [[binding[v] for v in res.vars] for binding in res.bindings]
            if v_type == 'quantity':
                value = float(res[0][2].value)
                unit = res[0][1].value
            else:
                value = res[0][0].value
            value = ValueClass(v_type, value, unit)
            parsed_answer = str(value)
        elif parse_type == 'bool':
            parsed_answer = 'yes' if res else 'no'
        elif parse_type == 'pred':
            parsed_answer = str(res[0][0])
            parsed_answer = parsed_answer.replace('_', ' ')
        return parsed_answer
    except Exception:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--kb_path', required=True)
    parser.add_argument('--ttl_path', required=True)
    args = parser.parse_args()

    data = DataForSPARQL(args.kb_path)
    engine = SparqlEngine(data, args.ttl_path)
