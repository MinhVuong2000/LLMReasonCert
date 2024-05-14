from SPARQLWrapper import SPARQLWrapper, JSON
import re
import json
from tqdm import tqdm
from generative_cert.utils.utils import processed_groundtruth_path


# https://github.com/lanyunshi/Multi-hopComplexKBQA/blob/master/code/SPARQL_test.py
# https://juejin.cn/post/7283690681175113740
# python virtuoso.py start 3001 -d virtuoso_db
# python virtuoso.py stop 3001
class SparQL(object):
    def __init__(self, SPARQLPATH):
        self.SPARQLPATH = SPARQLPATH
        self.sparql = SPARQLWrapper(
            SPARQLPATH
        )  #    PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>
        self.sparql.setReturnFormat(JSON)

    def query(self, sparql_txt, variable="value"):
        self.sparql.setQuery(sparql_txt)
        results = self.sparql.query().convert()
        id_vals = [
            str(v[variable]["value"]).split("/ns/")[-1]
            for v in results["results"]["bindings"]
        ]
        name_vals = [self.SQL_entity2name(id) for id in id_vals]
        return id_vals, name_vals

    def query_object(self, subject, rel):
        sparql_txt = (
            """PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT ?t WHERE {FILTER (!isLiteral(?t) OR lang(?t) = '' OR langMatches(lang(?t), 'en'))\nns:%s ns:%s ?t}"""
            % (subject, rel)
        )
        self.sparql.setQuery(sparql_txt)
        results = self.sparql.query().convert()
        try:
            obj_id = results["results"]["bindings"][0]["t"]["value"].split("/ns/")[-1]
        except:
            return None, None
        obj_name = self.SQL_entity2name(obj_id)
        sub_name = self.SQL_entity2name(subject)
        return [sub_name, rel, obj_name], obj_id

    def query_reasoning_path(self, sparql_txt):
        if "PREFIX rdf" in sparql_txt:  # grail_qa
            # sparql_txt = sparql_txt.replace('PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> ','')
            sparql_txt = re.sub("SELECT .+ WHERE", "CONSTRUCT WHERE", sparql_txt)
            sparql_txt = sparql_txt.replace("} }", "}").replace("}  }", "}")
        else:  # cwq, webqsp
            sparql_txt = re.sub(
                "SELECT DISTINCT .+\\nWHERE", "CONSTRUCT WHERE", sparql_txt
            )
            sparql_txt = sparql_txt.replace("#MANUAL SPARQL", "")
        # print('sparql_txt', sparql_txt)
        self.sparql.setQuery(sparql_txt)
        results = self.sparql.query().convert()
        # print(results)
        path = []
        for e in results["results"]["bindings"]:
            try:
                l = [str(v["value"]).split("/ns/")[-1] for v in e.values()]
                if l[1] == "type.object.type":
                    continue
                dic = [self.SQL_entity2name(id) for id in l]
                path.append(dic)
            except:
                print([v["value"] for v in e.values()])
                raise ValueError()
            # print(dic)
        processed_path = processed_groundtruth_path(path)
        # for tri in processed_path:
        #     print(tri)
        return path, processed_path

    def get_ents(self, sparql_txt):
        # retrieve all entities
        sparql_txt = sparql_txt.replace("DISTINCT ?x", "DISTINCT *")
        self.sparql.setQuery(sparql_txt)
        results = self.sparql.query().convert()
        # print(results)
        ent_l = []
        for e in results["results"]["bindings"]:
            dic = {k: v["value"].split("/ns/")[-1] for k, v in e.items()}
            dic = {code: self.SQL_entity2name(id) for code, id in dic.items()}
            ent_l.append(dic)
        return ent_l

    def SQL_query2path(self, sparql_txt):
        code2id_ents = self.get_ents(sparql_txt)
        # print(code2id_ents)
        # print()
        query_triplets = self.extract_triples_from_query(sparql_txt)
        # print(query_triplets)
        sparql_paths = []
        for dic in code2id_ents:
            path = [
                [dic.get(t[0], t[0]), t[1], dic.get(t[2], t[2])] for t in query_triplets
            ]
            sparql_paths.append(path)
        return sparql_paths

    def triplet_reverse(self, tri):
        head, rel, tail = tri
        head_condition = re.search("^[mg]\.", head) or re.search("^\d+", head)
        tail_condition = re.search("^[mg]\.", tail) or re.search("^\d+", tail)
        if head_condition and tail_condition:
            return tri
        if not head_condition:
            condition = "?t ns:%s ns:%s" % (rel, tail)
        else:
            condition = "ns:%s ns:%s ?t" % (head, rel)
        # print(condition)
        self.sparql.setQuery(
            """PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT ?t WHERE {FILTER (!isLiteral(?t) OR lang(?t) = '' OR langMatches(lang(?t), 'en'))\n%s}"""
            % condition
        )
        results = self.sparql.query().convert()
        # print(results)
        name = results["results"]["bindings"][0]["t"]["value"].split("/ns/")[-1]
        if not head_condition:
            return name, rel, tail
        else:
            return head, rel, name

    def SQL_entity2name(self, e):
        if not re.search("^[mg]\.", e):
            return e
        self.sparql.setQuery(
            """PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT ?t WHERE {FILTER (!isLiteral(?t) OR lang(?t) = '' OR langMatches(lang(?t), 'en'))\nns:%s ns:type.object.name ?t.}"""
            % (e)
        )
        try:
            results = self.sparql.query().convert()
            # print(results)
            name = (
                results["results"]["bindings"][0]["t"]["value"]
                if results["results"]["bindings"]
                else e
            )
        except:
            name = e
        return name

    def extract_triples_from_query(self, query_str):
        pattern = r"(ns:|\?).+ ns:.+ (ns:|\?).+ "
        query_l = query_str.split("\n")
        triplets = [re.search(pattern, q) for q in query_l]
        triplets = [tri.group().strip() for tri in triplets if tri]
        triplets = [tri.replace("ns:", "").replace("?", "") for tri in triplets]
        triplets = [tri.split(" ") for tri in triplets]
        # find id2name in case: not code
        triplets = [
            [self.SQL_entity2name(t[0]), t[1], self.SQL_entity2name(t[2])]
            for t in triplets
        ]
        return triplets


if __name__ == "__main__":
    SPARQLPATH = "http://localhost:3001/sparql"
    sparql = SparQL(SPARQLPATH)

    sparql_txt = 'PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.05kkh)\nFILTER (!isLiteral(?x) OR lang(?x) = \'\' OR langMatches(lang(?x), \'en\'))\nns:m.05kkh ns:government.governmental_jurisdiction.governing_officials ?y .\n?y ns:government.government_position_held.office_holder ?x .\n?y ns:government.government_position_held.basic_title ns:m.0fkvn .\nFILTER(NOT EXISTS {?y ns:government.government_position_held.from ?sk0} || \nEXISTS {?y ns:government.government_position_held.from ?sk1 . \nFILTER(xsd:datetime(?sk1) <= "2011-12-31"^^xsd:dateTime) })\nFILTER(NOT EXISTS {?y ns:government.government_position_held.to ?sk2} || \nEXISTS {?y ns:government.government_position_held.to ?sk3 . \nFILTER(xsd:datetime(?sk3) >= "2011-01-01"^^xsd:dateTime) })\n?x ns:government.politician.government_positions_held ?c .\n?c ns:government.government_position_held.from ?num .\nFILTER (?num < "1983-01-03"^^xsd:dateTime) . \n}'
    print(sparql.SQL_query2path(sparql_txt))
    print(
        sparql.triplet_reverse(
            ["m.0340r0", "government.politician.government_positions_held", "sk2"]
        )
    )  #'m.0mth_2g','government.government_position_held.to'
    print(sparql.SQL_entity2name("m.0bfmhy4"))
    print(sparql.query_reasoning_path(sparql_txt))
    # print()
    sparql_txt = "#MANUAL SPARQL\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\n\tFILTER (?x != ns:m.0d05w3)\n\t{ \n\t  ns:m.0d05w3 ns:location.statistical_region.places_exported_to ?y .\n\t  ?y ns:location.imports_and_exports.exported_to ?x . \n\t}\n\tUNION\n\t{\n\t  ns:m.0d05w3 ns:location.statistical_region.places_imported_from ?y .\n\t  ?y ns:location.imports_and_exports.imported_from ?x . \n\t}?x ns:location.location.time_zones ns:m.03bdv . \n}"
    paths = sparql.query_reasoning_path(sparql_txt)[1]
    print(paths, len(paths))

    sparql_txt = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> SELECT (?x0 AS ?value) WHERE { SELECT DISTINCT ?x0 WHERE { ?x0 :type.object.type :language.language_writing_type . ?x1 :type.object.type :language.language_writing_system . {SELECT (MAX(?y2) AS ?x2)  WHERE { ?x0 :language.language_writing_type.writing_systems ?x1 . ?x1 :language.language_writing_system.used_from ?x2 . FILTER ( ?x0 != ?x1 && ?x0 != ?x2 && ?x1 != ?x2  ) } }"
    # sparql_txt = re.sub(r' FILTER \( \?y.+ \?y\d *\)','', sparql_txt)
    paths = sparql.query_reasoning_path(sparql_txt)[1]
    print(paths, len(paths))

    print(sparql.query(sparql_txt, variable="value"))
    print(
        sparql.query_object(
            "m.02zb0l", "user.patrick.default_domain.warship_v1_1.commissioned"
        )
    )
