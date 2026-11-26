"""Build and audit the knowledge graph against the same passages used for retrieval."""

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from datetime import date
from hashlib import sha256
from pathlib import Path
import re
import unicodedata

from pydantic import BaseModel, Field

from .corpus import ChunkId, Label, Relation


SCHEMA_PATH = Path(__file__).with_name("graph_schema.json")
METER = re.compile(r"(?<![a-z0-9/／])([1-9][0-9]?)\s*[/／]\s*(1|2|4|8|16|32|64)(?![a-z0-9/／])", re.I)
ASCII_WORD = r"[a-zA-Z0-9]"


def normalize(label):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", label).casefold())


class Entity(BaseModel):
    id: ChunkId
    name: Label
    kind: Label = "concept"
    aliases: list[Label] = Field(default_factory=list)
    broad: bool = False


class Predicate(BaseModel):
    id: ChunkId
    name: Label
    legacy_labels: list[Label] = Field(default_factory=list)
    head_kinds: set[Label] = Field(default_factory=set)
    tail_kinds: set[Label] = Field(default_factory=set)
    exclusive: bool = False


class Rule(BaseModel):
    predicate: ChunkId
    pattern: str


class GraphSchema(BaseModel):
    entities: list[Entity]
    predicates: list[Predicate]
    rules: list[Rule]


@dataclass(frozen=True)
class Mention:
    entity_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Claim:
    head: str
    predicate: str
    tail: str
    evidence_id: str
    scope: str | None = None
    valid_from: date | None = None
    valid_to: date | None = None
    origins: tuple[str, ...] = ("reviewed",)
    spans: tuple[tuple[int, int], ...] = ()

    def key(self):
        return (self.head, self.predicate, self.tail, self.evidence_id, self.scope, self.valid_from, self.valid_to)


class KnowledgeGraph:
    def __init__(self, rows, reviewed, schema=None):
        schema = schema or GraphSchema.model_validate_json(SCHEMA_PATH.read_text(encoding="utf-8-sig"))
        self.entities = {}
        self.aliases = {}
        self.entity_docs = defaultdict(set)
        self.mentions = {}
        self.claims = {}
        self.rejected = []
        self.predicates = {}
        self.predicate_aliases = {}
        for entity in schema.entities:
            self.register(entity)
        for predicate in schema.predicates:
            if predicate.id in self.predicates:
                raise ValueError(f"重复的关系类型 ID：{predicate.id}")
            self.predicates[predicate.id] = predicate
            for name in [predicate.id, predicate.name, *predicate.legacy_labels]:
                key = normalize(name)
                if self.predicate_aliases.setdefault(key, predicate.id) != predicate.id:
                    raise ValueError(f"关系名称存在歧义：{name}")
        rules = [(rule.predicate, re.compile(rule.pattern, re.I)) for rule in schema.rules]
        for predicate, pattern in rules:
            if predicate not in self.predicates or not {"head", "tail"} <= pattern.groupindex.keys():
                raise ValueError(f"关系抽取规则无效：{predicate}。")
        for row in rows.values():
            for meter in METER.finditer(row["text"]):
                numerator, denominator = meter.groups()
                name = f"{numerator}/{denominator}"
                if not self.resolve(name):
                    self.register(Entity(id=f"meter-{numerator}-{denominator}", name=name, kind="meter"))
        for row in rows.values():
            for label in row["entities"]:
                self.register_label(label)
        for item in reviewed:
            relation = item if isinstance(item, Relation) else Relation(
                head=item[0], relation=item[1], tail=item[2], evidence_id=item[3])
            if relation.evidence_id not in rows:
                raise ValueError(f"不存在的证据段落：{relation.evidence_id}")
            self.register_label(relation.head)
            self.register_label(relation.tail)
            self.add_claim(relation.head, relation.relation, relation.tail, relation.evidence_id,
                           relation.scope, relation.valid_from, relation.valid_to)
        self.pattern = self.compile_ner()
        for row in rows.values():
            key = row["id"]
            mentions = set(self.find_mentions(row["text"]))
            for predicate, pattern in rules:
                for match in pattern.finditer(row["text"]):
                    claim = self.add_claim(match["head"], predicate, match["tail"], key,
                                           origins=("rule",), spans=(match.span("head"), match.span("tail")))
                    if claim:
                        for group, entity in (("head", claim.head), ("tail", claim.tail)):
                            mentions.add(Mention(entity, *match.span(group), match[group]))
            self.mentions[key] = sorted(mentions, key=lambda m: (m.start, m.end, m.entity_id))
            for entity in {m.entity_id for m in mentions} | {self.resolve(label) for label in row["entities"]}:
                self.entity_docs[entity].add(key)
        self.conflicts, quarantined = self.detect_conflicts()
        self.edges = defaultdict(list)
        for key, claim in self.claims.items():
            if key not in quarantined:
                self.edges[claim.head].append(claim)

    def register(self, entity):
        if entity.id in self.entities:
            raise ValueError(f"重复的实体 ID：{entity.id}")
        self.entities[entity.id] = entity
        for name in [entity.id, entity.name, *entity.aliases]:
            key = normalize(name)
            if not key or self.aliases.setdefault(key, entity.id) != entity.id:
                raise ValueError(f"实体别名存在歧义：{name}")

    def resolve(self, label):
        return self.aliases.get(normalize(label))

    def register_label(self, label):
        if not self.resolve(label):
            identifier = sha256(normalize(label).encode()).hexdigest()[:24]
            self.register(Entity(id=f"term-{identifier}", name=label))

    def compile_ner(self):
        spellings = {term for e in self.entities.values() if e.kind != "value" for term in (e.name, *e.aliases)}
        patterns = []
        for term in sorted(spellings, key=lambda text: (-len(text), text)):
            expr = r"\s+".join(re.escape(part) for part in term.split())
            if re.match(ASCII_WORD, term):
                expr = rf"(?<!{ASCII_WORD})" + expr
            if re.search(ASCII_WORD + "$", term):
                expr += rf"(?!{ASCII_WORD})"
            patterns.append(expr)
        return re.compile("|".join(patterns) if patterns else r"(?!)", re.I)

    def find_mentions(self, text):
        matches = [m for m in self.pattern.finditer(text)]
        # Meter matching also accepts spaces and full-width slashes in the source.
        matches.extend(METER.finditer(text))
        mentions = {}
        for match in sorted(matches, key=lambda m: m.start()):
            surface = match[0]
            identifier = self.resolve(surface)
            if identifier:
                key = (identifier, *match.span())
                mentions.setdefault(key, Mention(*key, surface))
        return list(mentions.values())

    def add_claim(self, head, relation, tail, evidence_id, scope=None, valid_from=None, valid_to=None,
                  origins=("reviewed",), spans=()):
        head_id, tail_id = self.resolve(head), self.resolve(tail)
        predicate = self.predicates.get(self.predicate_aliases.get(normalize(relation)))
        if (not predicate or not head_id or not tail_id
                or (predicate.head_kinds and self.entities[head_id].kind not in predicate.head_kinds)
                or (predicate.tail_kinds and self.entities[tail_id].kind not in predicate.tail_kinds)):
            self.rejected.append({"head": head, "relation": relation, "tail": tail, "evidence_id": evidence_id,
                                  "reason": "unknown_entity_or_predicate_or_invalid_direction"})
            return
        claim = Claim(head_id, predicate.id, tail_id, evidence_id, scope, valid_from, valid_to, origins, spans)
        previous = self.claims.get(claim.key())
        if previous:
            claim = replace(claim, origins=tuple(sorted(set(claim.origins) | set(previous.origins))),
                            spans=claim.spans or previous.spans)
        self.claims[claim.key()] = claim
        return claim

    def detect_conflicts(self):
        by_property = defaultdict(list)
        quarantined = set()
        conflicts = []
        for claim in self.claims.values():
            if self.predicates[claim.predicate].exclusive:
                key = (claim.head, claim.predicate, normalize(claim.scope) if claim.scope else None)
                for other in by_property[key]:
                    if (claim.tail != other.tail
                            and max(claim.valid_from or date.min, other.valid_from or date.min)
                            < min(claim.valid_to or date.max, other.valid_to or date.max)):
                        conflicts.append((other, claim))
                        quarantined.update((other.key(), claim.key()))
                by_property[key].append(claim)
        return conflicts, quarantined

    def expand(self, query, *, max_hops=2, max_edges=30):
        found = {m.entity_id for m in self.find_mentions(query)
                 if self.entities[m.entity_id].kind != "value"}
        specific = {key for key in found if not self.entities[key].broad}
        seeds = specific or found
        pending = deque((entity, []) for entity in sorted(seeds))
        visited = set(seeds)
        paths = {}
        checked = 0
        while pending and checked < max_edges:
            head, path = pending.popleft()
            if len(path) >= max_hops:
                continue
            for claim in self.edges.get(head, ()):
                if checked == max_edges:
                    break
                checked += 1
                label = self.predicates[claim.predicate].name
                step = [self.entities[head].name, label, self.entities[claim.tail].name, claim.evidence_id]
                next_path = [*path, step]
                paths.setdefault(claim.evidence_id, next_path)
                if claim.tail not in visited:
                    visited.add(claim.tail)
                    pending.append((claim.tail, next_path))
        return paths

    def counts(self):
        active = [claim for edges in self.edges.values() for claim in edges]
        return {"entities": len(self.entities), "mentions": sum(map(len, self.mentions.values())),
                "claims": len(self.claims), "active_claims": len(active),
                "rule_claims": sum("rule" in claim.origins for claim in active),
                "conflicts": len(self.conflicts), "rejected": len(self.rejected)}

    def report(self):
        return {
            "counts": self.counts(),
            "entities": [e.model_dump() for e in self.entities.values()],
            "predicates": [p.model_dump(mode="json") for p in self.predicates.values()],
            "mentions": {chunk: [asdict(m) for m in mentions] for chunk, mentions in self.mentions.items()},
            "entity_passages": {key: sorted(self.entity_docs[key]) for key in sorted(self.entity_docs)},
            "claims": [asdict(claim) for edges in self.edges.values() for claim in edges],
            "conflicts": [[asdict(claim) for claim in pair] for pair in self.conflicts],
            "rejected": self.rejected,
        }
