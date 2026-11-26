"""Reproduce the reviewed import of the 78-page Rosine print edition.

Run from the project root with Python, pypdf, pdfplumber, Pillow, and pydantic.
Pass the Poppler pdftoppm executable to --pdftoppm if it is not on PATH.
"""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess

import pdfplumber
from PIL import Image
from pypdf import PdfReader

from deepvocal.corpus import Chunk, Document, Figure, Relation
from deepvocal.ingest import write_document


DOC_ID = "rosine-vocal-techniques"
SHA256 = "5c74cae0585a50af4c7f44d6a7949fa601a359d741c96db602fce14060f20bfc"
TITLE = "Vocal Techniques for the Instrumentalist"
EDITION = "Amy Rosine，2021"
FIGURE_CREDIT = "出自 Amy Rosine，2021 年印刷版。教材采用 CC BY-NC 4.0，另有注明的内容除外。"
CHAPTERS = {
    7: "导论 / Introduction", 9: "歌唱机能 / The Vocal Instrument", 18: "歌唱动机 / Motivation",
    24: "呼吸 / Respiration", 28: "发声 / Phonation", 33: "共鸣 / Resonance",
    38: "咬字与 IPA / Articulation", 49: "歌曲学习与演唱 / Learning and Performing Vocal Solos",
    53: "嗓音类型与声区 / Identifying Voice Type", 56: "青少年变声 / Adolescent Singers",
    66: "附录 I · 练声谱例 / Vocal Exercises", 67: "附录 II · 嗓音健康 / Vocal Health",
    69: "附录 III · 器乐学习者的歌唱 / Singing Instrumentalists",
}
PAGES = [7, *range(9, 23), *range(24, 64), *range(66, 77)]
# Areas indexed separately as visual evidence, in PDF points with a top-left origin.
VOWEL_BOX = [(43, (54, 302, 342, 542)), (44, (54, 55, 342, 127))]
CONSONANT_BOX = [(47, (54, 510, 342, 554)), (48, (54, 56, 342, 127))]

# Page, a text-layer anchor, stable id suffix, Chinese evidence notes, reviewed outgoing edges.
NOTES = [
    (13, "A balanced, free, flexible posture", "alignment",
     "教材建议采用平衡、放松、有弹性的歌唱姿势，站立时不锁膝，并放松肩颈。",
     [("歌唱姿势", "涉及", "歌唱发声")]),
    (21, "Sing the vocal exercises from Appendix I", "score-guide",
     "教材指导读者练附录 I 的第 1 条谱例。读完每个句子后，用自己感受到的情绪演唱，再比较不同表达下的声音。",
     [("情绪表达练习", "练习示例", "练声谱例")]),
    (22, "progressively move from conversational speaking", "speech-to-singing",
     "把一句有情绪的日常话语从自然说话逐渐过渡到带音高的歌唱。注意气息，以及身体是否出现多余的紧张。",
     [("从说话过渡到歌唱", "关联观察", "音高")]),
    (24, "Keeping the spine long", "breath-alignment",
     "呼吸章把身体对齐与吸气时的胸廓运动联系起来。保持脊柱舒展，有助于肋骨运动和肺部扩张。",
     []),
    (27, "Inhale 4 counts", "breath-counts",
     "呼吸练习示例为吸气 4 拍、停留 2 拍、呼气 4 拍，并强调吸气轻松、稳定。原文还列了延长呼气的变化。",
     []),
    (28, "Sound is initiated by air passing", "voice-source",
     "气流通过喉内的声带开始发声，随后声音在声道中受到修饰。",
     [("发声", "涉及", "声带"), ("声带", "涉及", "喉部"), ("发声", "涉及", "声道")]),
    (29, "As air flows through the larynx", "vocal-fold-vibration",
     "气流经过喉部时声带振动，声音由声道修饰并放大。书中说明喉的位置会改变声道形状和音色。",
     [("喉部", "相关主题", "声道")]),
    (30, "they abduct (open)", "glottis",
     "歌唱发声涉及声带相合。教材把声带为呼吸而分开时形成的间隙称为声门，并描述声带随音高的伸长、变薄或放松、变厚。",
     [("声带", "相关主题", "声门")]),
    (30, "the other set of muscles are the cricothyroid muscles", "cricothyroid",
     "教材在发声肌肉一节介绍环甲肌，并描述声带拉长、变薄与较高音高的关系。",
     [("环甲肌", "相关主题", "声带")]),
    (31, "Too much air passing through the vocal folds", "breathy-tone",
     "教材把过多气流通过声带与偏漏气的气声音色联系起来。",
     [("气声", "涉及", "声带")]),
    (31, "Too much sub-glottal air pressure", "pressed-tone",
     "教材指出过多声门下气压可能使音色紧挤，并提醒不要用力把声音推出来。",
     [("挤压音色", "涉及", "声带")]),
    (31, "Trill or buzz the lips on one tone", "lip-trill",
     "唇颤音先在单一音高上做唇部振动，再上下滑动。做不出来时，教材建议试试像发冷一样说 brr，也可改做舌颤或哼鸣。",
     [("发声", "关联练习", "唇颤音"), ("唇颤音", "可用方法", "哼鸣")]),
    (32, "Sing “ah” from high to low", "sigh-glide",
     "从高到低唱轻松、连贯的 ah，试试 wee 或 oo 也可以。教材把它写成有控制的叹息滑音，提示不要一下把气推出去，喉部应轻松。",
     [("发声", "关联练习", "叹息滑音")]),
    (32, "whine like a puppy", "puppy-whine",
     "小狗呜咽练习用少量力气模仿呜咽，找到轻松而清晰的声音，再试着把这种感觉带到元音上。",
     [("发声", "关联练习", "小狗呜咽练习")]),
    (33, "A resonator is any object", "resonance",
     "共鸣体会过滤、放大或改变振动。改变共鸣腔的形状和大小会影响音色，教材用元音的形成说明这一点。",
     []),
    (36, "The soft palate muscles lower and raise", "soft-palate",
     "软腭的升降影响通往鼻腔的通道。教材以英语 m、n、ng 为例，说明这些鼻音会用到软腭下降。",
     [("软腭", "相关主题", "咬字")]),
    (42, "IPA is a worldwide standardized system", "ipa",
     "教材用国际音标（IPA）记录语音，辅助歌唱时的元音和辅音发音。",
     [("咬字", "涉及", "国际音标")]),
    (53, "There are six basic voice types", "voice-type",
     "作者区分古典独唱的声部分类与合唱声部分配。教材列出的古典类别包括女高音、女中音、女低音、男高音、男中音和男低音。",
     []),
    (54, "Voice type is determined mostly by", "comfortable-voice-type",
     "判断嗓音类型主要观察在哪个音区唱得舒服且有共鸣。初学者的音域可能随发育和练习变化。",
     [("嗓音类型", "涉及", "舒适音区")]),
    (54, "A piano is the best instrument", "vocal-range",
     "用钢琴测歌唱音域时，教材建议唱上行、下行五音音阶，可选 ah 或其他舒适元音，找到能舒适唱出的最高音与最低音。原文指向第 14 条谱例。",
     [("歌唱音域", "练习示例", "练声谱例")]),
    (55, "Common terms for vocal registers", "falsetto",
     "按照本教材的术语，假声（falsetto）指男性极轻、偏高音区所用的称呼。教材正文以轻机制和重机制讨论声区。",
     [("假声", "属于", "声区"), ("声区", "包含", "轻机制"), ("声区", "包含", "重机制")]),
    (55, "The area where you may feel a voice", "passaggio",
     "换声区（passaggio）是一个声区转到另一个声区时的过渡地带。肌肉会在过渡中调整，唱者可能感觉声音有断点或变化。",
     [("换声区", "过渡涉及", "声区")]),
    (59, "Use all vowels, starting the vocalese in light mechanism", "adolescent-exercise",
     "教材建议青少年用各种元音做发声练习，以适中音量从轻机制平稳向下过渡到重机制。",
     [("青少年变声", "涉及", "轻机制"), ("青少年变声", "涉及", "重机制")]),
    (59, "or falsetto pushed too low", "adolescent-falsetto",
     "青少年训练这一节提醒，不要把假声硬压向过低音高。",
     [("青少年变声", "涉及", "假声")]),
]


def join_lines(lines):
    text = ""
    for line in lines:
        text += (" " if text and not text.endswith("-") else "") + line
    return text


def page_sections(page, page_number):
    """Read the text layer in order, keeping paragraphs, lists, and exercise boundaries."""
    left = page.width * 0.143
    right = page.width - left
    heading = ""
    section = ""
    block = []
    previous_top = 0
    previous_left = 0
    for line in page.extract_text_lines(x_tolerance=1, return_chars=True):
        top, x, end = line["top"], line["x0"], line["x1"]
        if top > 558 or (page_number == 29 and x > 200 and 170 < top < 380):
            continue
        if ((page_number == 43 and top >= 302) or (page_number == 44 and top <= 127)
                or (page_number == 47 and top >= 510) or (page_number == 48 and top <= 127)):
            continue
        text = line["text"].strip()
        font, size = Counter((c["fontname"], c["size"]) for c in line["chars"] if c["text"].strip()).most_common(1)[0][0]
        if page_number in CHAPTERS and top < 125 and size >= 16:
            continue
        heading_line = "Cormorant" in font and size >= 10 or (
            "Lora-Bold" in font and len(text) < 60 and text[0].isalpha() and not text.endswith((".", ",", ":"))
        )
        if heading_line:
            if block:
                yield section, join_lines(block)
                block = []
            heading = f"{heading} {text}".strip() if heading and top - previous_top < 34 else text
            section = heading
        else:
            heading = ""
            # The typefaces use consistent indent for new paragraphs and list items.
            list_marker = r"^(?:•|\d+[.)]\s|\[[^]]+\]\s)"
            indented = (
                6 < x - previous_left <= 13 and (end >= right - 28 or x < left + 12)
                and not re.match(r"^[\[(/]", text)
                and not (block and re.match(list_marker, block[0]))
            )
            list_item = re.match(list_marker, text)
            if block and (top - previous_top > 19.8 or indented or list_item):
                yield section, join_lines(block)
                block = []
            block.append(text)
        previous_top, previous_left = top, x
    if block:
        yield section, join_lines(block)


def figure_chunk(page, end, suffix, section, text, caption, description, credit=FIGURE_CREDIT):
    return Chunk(id=f"rosine-p{page:03d}-{suffix}", section=section, text=text, page=page,
                 page_label=str(page - 6), page_end=end, page_end_label=str(end - 6) if end else None,
                 summary=description, figures=[Figure(id=suffix, caption=caption, description=description, credit=credit)])


def capture_composite(pdf_path, poppler, clips):
    dpi = 216
    thumbnails = []
    for page, bbox in clips:
        result = subprocess.run([poppler, "-f", str(page), "-l", str(page), "-r", str(dpi), "-png", "-singlefile",
                                 str(pdf_path)], check=True, capture_output=True)
        with Image.open(BytesIO(result.stdout)) as rendered:
            thumbnails.append(rendered.crop(tuple(round(pos * dpi / 72) for pos in bbox)).convert("RGB"))
    gap = 12
    canvas = Image.new("RGB", (max(frame.width for frame in thumbnails), sum(frame.height for frame in thumbnails) + gap), "white")
    top = 0
    for frame in thumbnails:
        canvas.paste(frame, (0, top))
        top += frame.height + gap
    return canvas


def figures(pdf_path, assets, poppler):
    reader = PdfReader(pdf_path)
    image_assets = {
        "larynx-front": next(iter(reader.pages[28].images)).image,
        "vocal-score": next(iter(reader.pages[65].images)).image,
        "ipa-vowels": capture_composite(pdf_path, poppler, VOWEL_BOX),
        "ipa-consonants": capture_composite(pdf_path, poppler, CONSONANT_BOX),
    }
    for key, data in image_assets.items():
        data.save(assets / f"{key}.png")

    return [
        (figure_chunk(29, None, "larynx-front", "发声 · 喉部正面图 / Front View of Larynx",
                      "Front View of Larynx. Labels: Hyoid Bone; Epiglottis; Thyrohyoid Membrane; Thyroid Cartilage; "
                      "Cricothyroid Ligament; Cricothyroid Muscles; Cricoid Cartilage; Trachea.",
                      "喉部正面图 / Front View of Larynx",
                      "喉部解剖正面图标出舌骨、会厌、甲状舌骨膜、甲状软骨、环甲韧带、环甲肌、环状软骨和气管。",
                      "图版出自 Amy Rosine 印刷版第 23 页。该页未单独署名。"),
         [("喉部", "相关主题", "喉部正面图"), ("喉部正面图", "图示", "舌骨"),
          ("喉部正面图", "图示", "会厌"), ("喉部正面图", "图示", "甲状软骨"),
          ("喉部正面图", "图示", "环状软骨")]),
        (figure_chunk(43, 44, "ipa-vowels", "咬字与 IPA · 开闭元音 / Vowel Classification",
                      "Vowel Classification\nWhen singing, open vowels use a taller space than closed.\n"
                      "Closed: [i] [e] [a] [o] [u]\nOpen: [ɪ] [ɛ] [ɑ] [ɔ] [ʊ]",
                      "开闭元音对照 / Vowel Classification",
                      "这张跨页 IPA 图表按教材列序对照闭元音 [i e a o u] 和开元音 [ɪ ɛ ɑ ɔ ʊ]。两行均可在引用内查看。"),
         [("国际音标", "涉及", "开元音"), ("国际音标", "涉及", "闭元音")]),
        (figure_chunk(47, 48, "ipa-consonants", "咬字与 IPA · 清浊辅音 / Voiced and Unvoiced Consonants",
                      "Voiced → Unvoiced\n[b] bust → [p] pool\n[d] ground → [t] grout\n[g] gown → [k] clown\n"
                      "[z] zoo → [s] school\n[ʃ] shoe ↔ [ʒ] garage\n[tʃ] choose ↔ [dʒ] just\n[v] voice → [f] fool",
                      "清辅音与浊辅音配对 / Voiced and Unvoiced Consonants",
                      "跨页辅音对照图列出七对发音方式相近的辅音，包括 [b] 与 [p]、[d] 与 [t]、[v] 与 [f]。原书有两对未遵循表头顺序。"),
         [("国际音标", "涉及", "浊辅音"), ("国际音标", "涉及", "清辅音")]),
        (figure_chunk(66, None, "vocal-score", CHAPTERS[66],
                      "Appendix I — Vocal Exercises. The original figure is a collection of numbered vocal exercises on staff notation. "
                      "Exercise 4: 'ah' from high to low. Exercise 8 works on onset. Exercise 14 is marked 'wee'. "
                      "See print page 15 for instructions on Exercise 1 and print page 48 for the reference to Exercise 14.",
                      "附录 I · 练声五线谱 / Vocal Exercises",
                      "附录 I 的练声五线谱在书内第 60 页。第 1 条可结合第 15 页的情绪表达练习，第 4 条从高到低唱 ah，第 8 条练起音，第 14 条唱 wee，并由第 48 页的音域练习引用。音符和唱词请查看图版。"),
         [("起音", "练习示例", "练声谱例"), ("练声谱例", "相关主题", "歌唱音域")]),
    ]


def extract_passages(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        sections = {page: list(page_sections(pdf.pages[page - 1], page)) for page in PAGES if page != 66}
    # Reconnect the exercise paragraph split by the printed page break.
    trill_index = next(i for i, (_, text) in enumerate(sections[31]) if "Trill or buzz the lips" in text)
    title, first = sections[31][trill_index]
    _, last = sections[32].pop(0)
    if not first.endswith("tongue-") or not last.startswith("trill (cat’s purr)"):
        raise ValueError("Lip trill continuation does not match the reviewed print edition.")
    sections[31][trill_index] = (title, join_lines([first, last]))
    return sections


def text_chunks(sections):
    notes = {page: [(anchor, suffix, summary, links) for p, anchor, suffix, summary, links in NOTES if p == page]
             for page in sections}
    matched = set()
    chapter = ""
    subheading = ""
    for page, blocks in sections.items():
        if page in CHAPTERS:
            chapter, subheading = CHAPTERS[page], ""
        grouped = []
        for heading, body in blocks:
            if heading:
                subheading = heading
            section = f"{chapter} · {subheading}" if subheading else chapter
            note = next((note for note in notes[page] if note[0].lower() in body.lower()), None)
            if note or not grouped or grouped[-1][0] != section or grouped[-1][2] or (
                len(grouped[-1][1]) + len(body) > 900
            ):
                grouped.append([section, body, note])
            else:
                grouped[-1][1] += "\n\n" + body
        for index, (section, body, note) in enumerate(grouped, 1):
            summary, edges = None, []
            suffix = f"{index:02d}"
            if note:
                _, suffix, summary, edges = note
                matched.add((page, suffix))
            yield Chunk(id=f"rosine-p{page:03d}-{suffix}", section=section[:160], text=body, summary=summary,
                        page=page, page_label=str(page - 6), page_end=32 if suffix == "lip-trill" else None,
                        page_end_label="26" if suffix == "lip-trill" else None), edges
    missing = [(page, suffix) for page, _, suffix, _, _ in NOTES if (page, suffix) not in matched]
    if missing:
        raise ValueError(f"Could not locate reviewed annotations: {missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--pdftoppm", default="pdftoppm")
    args = parser.parse_args()
    fingerprint = sha256(args.pdf.read_bytes()).hexdigest()
    if fingerprint != SHA256:
        parser.error("The reviewed annotations and crops require the attached 78-page print edition.")
    destination = args.data_dir / "knowledge" / f"{DOC_ID}.json"
    if destination.exists():
        parser.error(f"Corpus already exists: {destination}")
    assets = args.data_dir / "knowledge-assets" / DOC_ID
    assets.mkdir(parents=True, exist_ok=True)
    passages = [*text_chunks(extract_passages(args.pdf)), *figures(args.pdf, assets, args.pdftoppm)]
    chunks = []
    relations = []
    for chunk, links in sorted(passages, key=lambda item: item[0].page):
        for head, relation, tail in links:
            relations.append(Relation(head=head, relation=relation, tail=tail, evidence_id=chunk.id, scope=EDITION))
        # Use reviewed Chinese relation labels as local search keywords only when anchored to this evidence.
        chunk.entities = sorted({entity for head, _, tail in links for entity in (head, tail)})
        chunks.append(chunk)
    document = Document(id=DOC_ID, title=TITLE, edition=EDITION, source_url="https://doi.org/10.4148/npp.40721",
                        use="local", chunks=chunks, relations=relations)
    write_document(destination, document)
    print(json.dumps({"source_sha256": fingerprint, "pdf_pages": len(PdfReader(args.pdf).pages), "included_pages": len(PAGES),
                      "passages": len(chunks), "figures": sum(len(chunk.figures) for chunk in chunks),
                      "reviewed_relations": len(relations)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
