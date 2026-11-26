from collections import Counter
from hashlib import sha256
from math import log
import logging
import re

import numpy as np

from .corpus import load_corpus
from .graph import KnowledgeGraph
from .model import ModelError


log_service = logging.getLogger(__name__)
SEED_LIMIT = 8
MILVUS_SEARCH_TIMEOUT = 3
CROSS_MODES = ("cross_rerank", "cross_graph")

# One short, editable knowledge base. Every graph edge points back to a passage.
PASSAGES = [
    ("pitch", "音准与音分", "音准描述实际演唱频率与目标音高之间的偏差。负音分表示相对目标偏低，正音分表示偏高。先确认参考曲调和移调设置。", ["音准", "音分"]),
    ("flat", "音准偏低与跟唱", "单次音准偏低可以先用单音跟唱复查。听标准音，停顿后轻声唱同一个音，再对比两次录音。", ["音准", "音准偏低", "音高偏差", "跑调", "单音跟唱"]),
    ("sustain", "长音的音高稳定性", "长音稳定性可以观察去除起音和收音后的音高曲线。先用舒适音做短长音，再观察持续段是否逐渐偏移。颤音本身不等于不稳定。", ["长音稳定性", "音高波动", "颤音", "短长音练习"]),
    ("rhythm", "起音与节奏", "起音时间偏差是实际起音相对于参考音符开始时间的差。可以把困难节奏用慢速节拍器拍出来，再念节奏，最后加回旋律。", ["节奏", "起音偏差", "抢拍", "拖拍", "节拍器练习"]),
    ("register", "换声区", "换声区是声区转换可能变得明显的一段音域。可以在舒适音域先尝试轻声滑音，缩小范围，避免为了到达高音而增加音量。", ["换声区", "破音", "轻声滑音"]),
    ("falsetto-practice", "假声入门训练：轻声下滑与短下行琶音",
     "练习假声（falsetto）时，可从容易发声的较高音开始，轻声用 u（像‘呜’）在小音程内做连续下滑。"
     "能轻松、连贯地发声后，再试短下行琶音。专练换声区时，可在过渡点附近选两个相邻音，慢慢来回，观察音色和音量是否平顺，"
     "以及肩颈、下颌是否在用力。起点、音程和元音按个人的舒适范围调整。",
     ["假声", "换声区", "轻声滑音"]),
    ("thyroid-cartilage-muscles", "甲状软骨的肌肉附着与发声作用",
     "甲状软骨是喉部骨架的一部分。与声乐常相关的附着肌肉包括两类。"
     "喉内肌中，环甲肌连接环状软骨与甲状软骨，使声带拉长、张力增大，通常有助于升高音高。"
     "甲杓肌从甲状软骨内面连到杓状软骨，参与声带靠拢和缩短，它对音高的作用会随其他肌肉的活动而变。"
     "喉外肌中，胸骨甲状肌连接胸骨与甲状软骨，甲状舌骨肌连接甲状软骨与舌骨，两者参与调整喉的位置。"
     "甲状舌骨膜是连接甲状软骨和舌骨的结缔组织。",
     ["甲状软骨", "环甲肌", "甲杓肌", "胸骨甲状肌", "甲状舌骨肌", "声带"]),
    ("laryngeal-coordination-practice", "喉部发声协调训练：唇颤音和轻声滑音",
     "歌唱中与甲状软骨相关的环甲肌和甲杓肌共同参与音高调节。日常练声的目标是发声和音高转换的协调。"
     "可从舒适音区的短唇颤音或轻声哼鸣开始，再在小范围内缓慢做下行滑音，容易完成后再试上下滑动。"
     "观察声音是否连续、音量是否平稳、肩颈或下颌是否为了升高音高而使劲。保持轻松，感觉费力就缩小范围。"
     "这类练习无需用手按压或移动喉部，也无需刻意孤立某一块喉肌。",
     ["甲状软骨", "环甲肌", "甲杓肌", "声带", "发声", "唇颤音", "轻声滑音"]),
    ("lip", "唇颤音练习", "唇颤音是一项可用于低强度热身的练习。保持音量舒适，先练短音，再试小范围滑音。如果无法轻松完成，可以改做轻声哼鸣。", ["唇颤音", "热身", "轻声滑音", "哼鸣"]),
    ("breath", "短句分组", "当练习长乐句时，可以先标记计划换气的位置，用较慢速度分组练习，然后逐步把短句连起来。", ["长乐句", "换气", "短句分组"]),
    ("record", "录音条件", "为了让两次演唱测评容易比较，尽量使用同样的参考调、速度和录音距离。伴奏串音、背景噪声和削波可能影响分析。", ["录音质量", "参考调", "移调"]),
    ("phrases", "乐句拆分与复测", "标记困难小节，每次只修改一个因素，例如音高或节奏。用相同调性和速度复录，再比较同一段的变化。", ["短句分组", "复测", "节奏"]),
]
PASSAGE_SOURCES = {
    "falsetto-practice": (
        "Christopher Arneson · A Basic Guide to Vocal Exercises for Beginning Singing Teachers · 声区协调（中文教学摘要）",
        "https://www.rider.edu/sites/default/files/files/voiceteacher-basic_guide_for_beginning_teachers.pdf",
    ),
    "thyroid-cartilage-muscles": (
        "Modeling the influence of the extrinsic musculature on phonation · 第 2 节（中文摘要）",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC10529543/",
    ),
    "laryngeal-coordination-practice": (
        "Ingo Titze · The Five Best Vocal Warm-Up Exercises · Journal of Singing（教学整理）",
        "https://vocology.utah.edu/_resources/documents/titzes_top_five_vocal_warm_ups.pdf",
    ),
}
EDGES = [
    ("音分", "用于表示", "音准", "pitch"),
    ("音准", "常见偏差", "音准偏低", "flat"),
    ("音准偏低", "关联练习", "单音跟唱", "flat"),
    ("音高波动", "关联观察", "长音稳定性", "sustain"),
    ("长音稳定性", "关联练习", "短长音练习", "sustain"),
    ("起音偏差", "关联观察", "节奏", "rhythm"),
    ("节奏", "关联练习", "节拍器练习", "rhythm"),
    ("破音", "相关主题", "换声区", "register"),
    ("换声区", "关联练习", "轻声滑音", "register"),
    ("假声", "关联练习", "轻声滑音", "falsetto-practice"),
    ("甲状软骨", "有附着肌", "环甲肌", "thyroid-cartilage-muscles"),
    ("甲状软骨", "有附着肌", "甲杓肌", "thyroid-cartilage-muscles"),
    ("甲状软骨", "有附着肌", "胸骨甲状肌", "thyroid-cartilage-muscles"),
    ("甲状软骨", "有附着肌", "甲状舌骨肌", "thyroid-cartilage-muscles"),
    ("甲状软骨", "涉及", "声带", "thyroid-cartilage-muscles"),
    ("声带", "关联练习", "轻声滑音", "laryngeal-coordination-practice"),
    ("轻声滑音", "可用热身", "唇颤音", "lip"),
    ("长乐句", "关联练习", "短句分组", "breath"),
    ("短句分组", "可用方法", "复测", "phrases"),
    ("音准", "比较条件", "参考调", "record"),
]
EXERCISES = {
    "pitch_flat": ("flat", "单音跟唱", "听参考音，轻声模唱，然后播放录音比较。"),
    "pitch_sharp": ("flat", "单音跟唱", "听参考音，轻声模唱，然后播放录音比较。"),
    "timing": ("rhythm", "慢速节拍器", "先拍节奏，再按节拍念节奏，最后唱原旋律。"),
    "stability": ("sustain", "短长音", "在舒适音高保持短长音，比较持续段。"),
    "register": ("register", "轻声滑音", "在舒适范围做短滑音，保持轻声。"),
    "general": ("lip", "唇颤音热身", "先做短音，再尝试舒适的小范围滑音。"),
}


def terms(text):
    tokens = []
    for part in re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", text.lower()):
        if "\u3400" <= part[0] <= "\u9fff":
            tokens.extend(part)
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


def passage_search_text(row):
    title, body, entities = row["title"], row["text"], " ".join(row["entities"])
    annotations = [row.get("summary"), *(figure["description"] for figure in row.get("figures", []))]
    context = " ".join(value for value in annotations if value)
    if context:
        return f"{title} {context} {entities} {body}"
    return f"{title} {body} {entities}"


class Retriever:
    def __init__(self, uri=None, token=None, embed=None, embedding_id="", *, corpus_dir=None,
                 dimensions=None, local_embed=None, local_uri=None, reranker=None):
        self.rows = {
            key: dict(id=key, title=title, text=body, entities=entities, source="DeepVocal 示例知识库",
                      allow_cloud=True)
            for key, title, body, entities in PASSAGES
        }
        for key, (source, url) in PASSAGE_SOURCES.items():
            self.rows[key].update(source=source, source_url=url)
        imported, reviewed_edges = load_corpus(corpus_dir, structured_relations=True)
        standard_entities = {entity for _, _, _, entities in PASSAGES for entity in entities}
        for passage in imported:
            if passage["id"] in self.rows:
                raise ValueError(f"段落 ID 与示例证据重复：{passage['id']}")
            mentioned = {entity for entity in standard_entities if entity in passage["text"]}
            passage["entities"] = sorted(set(passage["entities"]) | mentioned)
            self.rows[passage["id"]] = passage
        self.imported_documents = len({row["document_id"] for row in imported})
        self.local_only_count = sum(not row["allow_cloud"] for row in imported)
        if self.local_only_count:
            # Keep both the text and its vectors on the host for a local corpus.
            uri, token = local_uri, None
            if local_embed:
                embed, embedding_id, dimensions = local_embed
            else:
                embed, embedding_id, dimensions = None, "", None
        self.keys = list(self.rows)
        texts = [passage_search_text(row) for row in self.rows.values()]
        df = Counter()
        for content in texts:
            df.update(set(terms(content)))
        vocabulary = sorted(df, key=lambda term: (-df[term], term))[:8192]
        self.term_index = {term: i for i, term in enumerate(vocabulary)}
        self.idf = {term: log((1 + len(texts)) / (1 + df[term])) + 1 for term in self.term_index}
        self.vectors = self.vectorize(texts)
        self.graph = KnowledgeGraph(self.rows, [*EDGES, *reviewed_edges])
        self.embed = embed
        self.dimensions = dimensions
        self.embedding_id = embedding_id if embed else None
        self.reranker = reranker
        self.client = None
        self.mode = "local_tfidf"
        if uri:
            self.build_index(uri, token, texts)

    def build_index(self, uri, token, texts):
        from pymilvus import MilvusClient

        options = {"grpc_options": {"grpc.keepalive_permit_without_calls": False}} if uri.endswith(".db") else {}
        self.client = MilvusClient(uri=uri, token=token or "", **options)
        try:
            if not self.embed:
                self.dimensions = len(self.term_index)
            lexical_schema = self.idf if not self.embed else None
            schema = (2, self.keys, texts, lexical_schema, self.embedding_id, self.dimensions)
            digest = sha256(repr(schema).encode()).hexdigest()[:20]
            self.collection = f"deepvocal_{digest}"
            exists = self.client.has_collection(self.collection)
            if not exists or int(self.client.get_collection_stats(self.collection)["row_count"]) != len(texts):
                vectors = self.ordered_embeddings(texts).tolist() if self.embed else self.vectors.tolist()
                if not exists:
                    self.client.create_collection(
                        collection_name=self.collection, dimension=len(vectors[0]),
                        metric_type="COSINE", consistency_level="Strong",
                    )
                records = [{"id": i, "vector": vector} for i, vector in enumerate(vectors)]
                for offset in range(0, len(records), 256):
                    self.client.upsert(self.collection, records[offset:offset + 256])
                self.client.flush(self.collection)
            self.client.load_collection(self.collection)
        except Exception:
            self.client.close()
            raise
        self.mode = "milvus_embedding" if self.embed else "milvus_tfidf"

    def ordered_embeddings(self, texts):
        try:
            vectors = np.asarray(self.embed(texts), dtype=np.float32)
            if (vectors.ndim != 2 or vectors.shape[0] != len(texts) or not vectors.shape[1]
                    or (self.dimensions and vectors.shape[1] != self.dimensions) or not np.isfinite(vectors).all()):
                raise ValueError
            return vectors
        except (TypeError, ValueError):
            raise ModelError("Embedding 向量维度或数值无效。") from None

    def vectorize(self, texts):
        vectors = np.zeros((len(texts), len(self.term_index)), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in Counter(terms(text)).items():
                if term in self.idf:
                    index = self.term_index[term]
                    vectors[row, index] += (1 + log(count)) * self.idf[term]
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norm, 1e-8)

    def search(self, query, mode="graph", top_k=3):
        lexical = self.vectorize([query])[0]
        lexical_scores = self.vectors @ lexical
        lexical_candidates = {self.keys[i]: float(lexical_scores[i]) for i in
                              np.argsort(-lexical_scores, kind="stable")[:max(SEED_LIMIT, top_k)]
                              if lexical_scores[i] > 0}
        seed_scores = lexical_candidates
        if self.client and (self.embed or lexical.any()):
            from grpc import FutureTimeoutError, RpcError
            from pymilvus.exceptions import MilvusException

            try:
                vector = self.ordered_embeddings([query])[0].tolist() if self.embed else lexical.tolist()
                hits = self.client.search(self.collection, data=[vector], limit=max(SEED_LIMIT, top_k),
                                          timeout=MILVUS_SEARCH_TIMEOUT)[0]
                seed_scores = {self.keys[int(hit["id"])]: float(hit["distance"])
                               for hit in hits if hit["distance"] > 0}
            except ModelError as exc:
                log_service.warning("向量模型查询失败，改用本地 TF-IDF。%s", exc)
            except (MilvusException, RpcError, FutureTimeoutError) as exc:
                log_service.warning("Milvus 查询失败，改用本地 TF-IDF（%s）。", type(exc).__name__)
        if mode == "vector":
            return [{**self.rows[key], "score": round(score, 4), "path": []}
                    for key, score in list(seed_scores.items())[:top_k]]
        paths = self.graph.expand(query) if mode in ("graph", "cross_graph") else {}
        lexical_by_id = dict(zip(self.keys, lexical_scores))
        pool = list(dict.fromkeys([*seed_scores, *lexical_candidates, *paths]))

        def score(key):
            graph_bonus = 0.04 / len(paths[key]) if key in paths else 0
            exact_matches = sum(entity in query for entity in self.rows[key]["entities"])
            return 0.55 * seed_scores.get(key, 0) + 0.45 * float(lexical_by_id[key]) + 0.06 * exact_matches + graph_bonus

        ranker = {}
        ranked = None
        if mode in CROSS_MODES:
            ranker = {"reranker": "rules"}
            if self.reranker and pool and top_k > 0 and all(self.rows[key]["allow_cloud"] for key in pool):
                documents = [f"{self.rows[key]['title']}\n{self.rows[key]['text']}" for key in pool]
                try:
                    ranked = [(pool[index], value) for index, value in self.reranker.rerank(query, documents, top_k)]
                    ranker = {"reranker": self.reranker.name}
                except ModelError as exc:
                    log_service.warning("重排模型调用失败，使用本地规则重排。%s", exc)
        if ranked is None:
            order = sorted(pool, key=lambda key: -score(key))[:top_k]
            ranked = [(key, score(key)) for key in order]
        return [
            {**self.rows[key], "score": round(value, 4), "path": paths.get(key, []), **ranker}
            for key, value in ranked
        ]
