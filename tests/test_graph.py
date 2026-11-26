from datetime import date
import unittest

from deepvocal.corpus import Relation
from deepvocal.graph import Entity, GraphSchema, KnowledgeGraph, Predicate, Rule


def passage(identifier, text):
    return {"id": identifier, "text": text, "entities": []}


class KnowledgeGraphTest(unittest.TestCase):
    def setUp(self):
        self.schema = GraphSchema(
            entities=[
                Entity(id="zhang-san", name="张三", kind="person", aliases=["小张"]),
                Entity(id="finance", name="财务部", kind="department", aliases=["财务部门"]),
                Entity(id="tech", name="技术部", kind="department"),
            ],
            predicates=[Predicate(id="member-of", name="隶属", head_kinds={"person"},
                                  tail_kinds={"department"}, exclusive=True)],
            rules=[Rule(predicate="member-of", pattern=r"(?P<head>小张|张三)目前隶属(?P<tail>财务部门|技术部)")],
        )

    def test_aliases_are_normalized_before_conflicts_are_isolated(self):
        rows = {"first": passage("first", "小张目前隶属财务部门。"),
                "second": passage("second", "张三目前隶属技术部。")}
        graph = KnowledgeGraph(rows, [], self.schema)

        self.assertEqual(graph.resolve("小张"), graph.resolve("张三"))
        self.assertEqual(graph.counts()["conflicts"], 1)
        self.assertEqual(graph.counts()["active_claims"], 0)
        self.assertEqual(graph.expand("小张在哪个部门？"), {})
        first, second = graph.conflicts[0]
        self.assertEqual((first.head, first.predicate, first.tail), ("zhang-san", "member-of", "finance"))
        self.assertEqual((second.head, second.predicate, second.tail), ("zhang-san", "member-of", "tech"))
        self.assertEqual([rows[first.evidence_id]["text"][start:end] for start, end in first.spans],
                         ["小张", "财务部门"])
        mention = graph.mentions["first"][0]
        self.assertEqual((mention.entity_id, mention.start, mention.end, mention.text), ("zhang-san", 0, 2, "小张"))

    def test_adjacent_intervals_and_other_scopes_are_compatible_but_reversed_edges_are_rejected(self):
        rows = {"old": passage("old", "张三曾任职于财务部。"),
                "new": passage("new", "张三调入技术部。"),
                "project": passage("project", "张三参与财务部的项目。")}
        changeover = date(2025, 1, 1)
        reviewed = [
            Relation(head="张三", relation="隶属", tail="财务部", evidence_id="old", scope="人事",
                     valid_to=changeover),
            Relation(head="小张", relation="member-of", tail="技术部", evidence_id="new", scope="人事",
                     valid_from=changeover),
            Relation(head="小张", relation="隶属", tail="财务部", evidence_id="project", scope="项目"),
            Relation(head="技术部", relation="隶属", tail="小张", evidence_id="new"),
        ]
        graph = KnowledgeGraph(rows, reviewed, self.schema)

        self.assertEqual(graph.counts()["conflicts"], 0)
        self.assertEqual(graph.counts()["rejected"], 1)
        self.assertEqual(set(graph.expand("小张")), {"old", "new", "project"})
        self.assertEqual(graph.expand("技术部在哪？"), {})

        reviewed[0] = reviewed[0].model_copy(update={"valid_to": date(2025, 2, 1)})
        overlapping = KnowledgeGraph(rows, reviewed, self.schema)
        self.assertEqual(overlapping.counts()["conflicts"], 1)
        self.assertEqual(set(overlapping.expand("张三")), {"project"})

    def test_music_rules_extract_new_meters_and_recognize_english_aliases(self):
        rows = {"meter": passage("meter", "12 ／ 16 常按每小节四个大拍。"),
                "subdivision": passage("subdivision", "6/8 每小节三拍可按另外的分组练习。"),
                "posture": passage("posture", "Check your singing posture. First, you can use a mirror.")}
        graph = KnowledgeGraph(rows, [])

        self.assertEqual(graph.counts()["rule_claims"], 2)
        self.assertEqual(graph.resolve("12 ／ 16"), "meter-12-16")
        self.assertEqual(graph.expand("12/16 有几个大拍？")["meter"],
                         [["12/16", "每小节大拍数", "4", "meter"]])
        self.assertEqual(graph.expand("唱歌姿势怎么自查？"), graph.expand("How is my singing posture?"))
        self.assertEqual(set(graph.expand("唱歌姿势怎么自查？")), {"posture"})
