"""Разбор текста редакции по оглавлению источника.

HTML редакции — плоский список абзацев <p id="pNNN">. Оглавление (getcontent) задаёт
структуру: часть кодекса → раздел → глава → статья → часть статьи / пункт / подпункт,
и для каждого узла — id первого и последнего абзаца (np, npe). Границы считаем
по позиции абзаца в документе: сами id идут не по порядку.

Редакционные пометки «(В редакции Федерального закона …)» в текст нормы не входят,
а сохраняются отдельно — по ним видно, какой закон изменил статью.
"""

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from lxml import html as lxml_html

ARTICLE_UNIT = "статья"
# span-пометки «(В редакции …)» и абзацы-пометки: о дополнении статьёй,
# об особенностях применения, общий перечень изменяющих законов
NOTE_SPANS = "mark", "markx"
NOTE_PARAGRAPHS = {"P", "A", "C"}
# надстрочный номер: «Статья 19<W9>1</W9>» — это статья 19.1
SUPERSCRIPT_SPAN = "W9"

# временная метка надстрочного индекса (Private Use Area — в текстах законов не встречается)
_SUP_MARK = ""
_SUP_TAG = re.compile(r"<sup>(.*?)</sup>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")
# номера вида 81, 56.1, 341.1-1 (в оглавлении — 341<sup>1-1</sup>)
_ARTICLE_CAPTION = re.compile(r"^Статья\s+(\d+(?:[.-]\d+)*)\.?\s*(.*)$", re.DOTALL)
_REPEALED = re.compile(r"^\(?\s*Статья утратила силу", re.IGNORECASE)


class ParseError(Exception):
    pass


@dataclass
class Fragment:
    """Часть статьи, пункт или подпункт: ref = [["часть", "1"], ["пункт", "6"]]."""

    ref: list[list[str]]
    text: str


@dataclass
class ParsedArticle:
    key: str
    order: int
    heading: str
    title: str
    path: list[str]
    text: str
    fragments: list[Fragment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    repealed: bool = False

    def content_hash(self) -> str:
        """Хеш содержания статьи: одинаковый текст в разных редакциях — одна версия."""
        payload = [self.heading, self.text, self.path, self.notes, self.repealed]
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class ParsedDocument:
    articles: list[ParsedArticle]
    warnings: list[str]


@dataclass
class _Paragraph:
    id: str | None
    text: str
    notes: list[str]


def normalize(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def _join_superscripts(text: str) -> str:
    # «19<метка>1» → «19.1»; надстрочный индекс не после цифры просто приклеиваем
    return re.sub(rf"(\d){_SUP_MARK}", r"\1.", text).replace(_SUP_MARK, "")


def caption_text(caption: str) -> str:
    """Подпись из оглавления в обычный текст: «Статья 56<sup>1</sup>. …» → «Статья 56.1. …»."""
    text = _SUP_TAG.sub(lambda m: _SUP_MARK + m.group(1), caption)
    return normalize(_join_superscripts(_ANY_TAG.sub("", text)))


def _paragraph(p: lxml_html.HtmlElement) -> _Paragraph:
    el = deepcopy(p)
    for sup in el.xpath(f".//span[@class='{SUPERSCRIPT_SPAN}']"):
        sup.text = _SUP_MARK + (sup.text or "")

    note_xpath = " or ".join(f"@class='{c}'" for c in NOTE_SPANS)
    notes = []
    for span in el.xpath(f".//span[({note_xpath}) and not(ancestor::span[{note_xpath}])]"):
        note = normalize(_join_superscripts(span.text_content()))
        if note:
            notes.append(note)
        span.drop_tree()  # хвостовой текст после пометки сохраняется

    text = normalize(_join_superscripts(el.text_content()))
    if p.get("class") in NOTE_PARAGRAPHS:
        notes, text = notes + ([text] if text else []), ""
    return _Paragraph(id=p.get("id"), text=text, notes=notes)


def _paragraphs(html: str) -> list[_Paragraph]:
    try:
        root = lxml_html.fromstring(html)
    except (ValueError, lxml_html.etree.ParserError) as exc:
        raise ParseError(f"HTML не разобран: {exc}") from exc
    body = root.find("body")
    if body is None:
        raise ParseError("в HTML нет <body>")
    return [_paragraph(p) for p in body.iter("p")]


def _span(
    node: dict[str, Any],
    index: dict[str, int],
    fallback_end: int | None,
    label: str,
    warnings: list[str],
) -> tuple[int, int] | None:
    start, end = index.get(node.get("np", "")), index.get(node.get("npe", ""))
    if start is None:
        warnings.append(f"{label}: начальный абзац {node.get('np')} не найден в тексте")
        return None
    if end is None or end < start:
        # в оглавлении источника встречаются ссылки на несуществующие абзацы —
        # тогда узел заканчивается перед началом следующего узла того же уровня
        if fallback_end is None or fallback_end < start:
            warnings.append(f"{label}: конечный абзац {node.get('npe')} не найден в тексте")
            return None
        warnings.append(
            f"{label}: конечный абзац {node.get('npe')} не найден, граница по следующему узлу"
        )
        end = fallback_end
    return start, end


def _next_start(
    nodes: list[dict[str, Any]], pos: int, index: dict[str, int], default: int | None
) -> int | None:
    """Абзац перед началом следующего узла того же или более высокого уровня."""
    lvl = nodes[pos].get("lvl", 0)
    for node in nodes[pos + 1 :]:
        if node.get("lvl", 0) <= lvl and node.get("np") in index:
            return index[node["np"]] - 1
    return default


def _unit_ref(caption: str) -> list[str]:
    # «$часть 1» → ["часть", "1"]; «$подпункт "а"» → ["подпункт", "а"]
    unit, _, num = caption_text(caption).lstrip("$").partition(" ")
    return [unit, num.strip().strip('"«»')]


def _article(
    node: dict[str, Any],
    children: list[dict[str, Any]],
    path: list[str],
    paragraphs: list[_Paragraph],
    index: dict[str, int],
    fallback_end: int | None,
    order: int,
    warnings: list[str],
) -> ParsedArticle | None:
    caption = caption_text(node.get("caption", ""))
    match = _ARTICLE_CAPTION.match(caption)
    if not match:
        warnings.append(f"не распознан номер статьи: {caption!r}")
        return None
    key, title = match.group(1), match.group(2).strip()

    bounds = _span(node, index, fallback_end, f"ст. {key}", warnings)
    if bounds is None:
        return None
    start, end = bounds

    heading = paragraphs[start].text or caption
    lines = [p.text for p in paragraphs[start + 1 : end + 1] if p.text]
    notes = [n for p in paragraphs[start : end + 1] for n in p.notes]
    # отметка «(Статья утратила силу …)» бывает и текстом, и пометкой
    repeal_note = next((x for x in lines[:1] + notes if _REPEALED.match(x)), None)
    repealed = node.get("state") == 1 or (len(lines) <= 1 and repeal_note is not None)
    if repealed and not lines and repeal_note:
        lines = [repeal_note]
    if not lines:
        warnings.append(f"ст. {key}: пустой текст")

    fragments: list[Fragment] = []
    stack: list[tuple[int, list[str]]] = []
    for pos, child in enumerate(children):
        lvl = child.get("lvl", 0)
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        stack.append((lvl, _unit_ref(child.get("caption", ""))))
        label = f"ст. {key}, {child.get('caption', '').lstrip('$')}"
        child_end = _next_start(children, pos, index, end)
        child_bounds = _span(child, index, child_end, label, warnings)
        if child_bounds is None:
            continue
        text = "\n".join(
            p.text for p in paragraphs[child_bounds[0] : child_bounds[1] + 1] if p.text
        )
        fragments.append(Fragment(ref=[ref for _, ref in stack], text=text))

    return ParsedArticle(
        key=key,
        order=order,
        heading=heading,
        title=title,
        path=path,
        text="\n".join(lines),
        fragments=fragments,
        notes=notes,
        repealed=repealed,
    )


def parse_redaction(html: str, toc: list[dict[str, Any]]) -> ParsedDocument:
    paragraphs = _paragraphs(html)
    index = {p.id: i for i, p in enumerate(paragraphs) if p.id}
    warnings: list[str] = []
    articles: list[ParsedArticle] = []
    seen: set[str] = set()
    path: list[tuple[int, str]] = []

    i = 0
    while i < len(toc):
        node = toc[i]
        lvl = node.get("lvl", 0)
        if node.get("unit") != ARTICLE_UNIT:
            # структурный узел вне статьи: часть кодекса, раздел, глава
            while path and path[-1][0] >= lvl:
                path.pop()
            path.append((lvl, caption_text(node.get("caption", ""))))
            i += 1
            continue

        j = i + 1
        while j < len(toc) and toc[j].get("lvl", 0) > lvl:
            j += 1
        ancestors = [caption for level, caption in path if level < lvl]
        fallback_end = _next_start(toc, i, index, len(paragraphs) - 1)
        article = _article(
            node,
            toc[i + 1 : j],
            ancestors,
            paragraphs,
            index,
            fallback_end,
            len(articles),
            warnings,
        )
        if article is not None:
            if article.key in seen:
                warnings.append(f"ст. {article.key}: повторяется в оглавлении, взята первая")
            else:
                seen.add(article.key)
                articles.append(article)
        i = j

    if not articles:
        raise ParseError("в редакции не найдено ни одной статьи")
    return ParsedDocument(articles=articles, warnings=warnings)
