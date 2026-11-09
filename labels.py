"""Extract full review sentences and direct attribute evidence."""
from __future__ import annotations
import re
from typing import Any
SCHEMA_VERSION = 'exp33-spans-1'
_CUES: tuple[tuple[str, str], ...] = (('soft', 'soft(?:ness|er|est)?'), ('silky', 'silk(?:y|ier|iest|en)|silk[- ]like'), ('satiny', 'satin(?:y|ier|iest)|satin[- ]like'), ('rough', 'rough(?:ness|er|est)?'), ('scratchy', 'scratch(?:y|ier|iest|iness)'), ('itchy', 'itch(?:y|ier|iest|iness)'), ('smooth', 'smooth(?:ness|er|est)?'), ('stiff', 'stiff(?:ness|er|est)?'), ('rigid', 'rigid(?:ity)?'), ('hard', 'hard(?:ness|er|est)?'), ('harsh', 'harsh(?:ness|er|est)?'), ('thin', 'thin(?:ness|ner|nest)?'), ('thick', 'thick(?:ness|er|est)?'), ('stretchy', 'stretch(?:y|ier|iest|iness)?'), ('elastic', 'elastic(?:ity)?'), ('breathable', 'breathab(?:le|ility)'), ('coarse', 'coarse(?:ness|r|st)?'), ('plush', 'plush(?:ness|er|est)?'), ('velvety', 'velvet(?:y|ier|iest)|velvet[- ]like'), ('fuzzy', 'fuzz(?:y|ier|iest|iness)'), ('fluffy', 'fluff(?:y|ier|iest|iness)'), ('furry', 'furr(?:y|ier|iest|iness)'), ('supple', 'supple(?:ness|r|st)?'), ('pliable', 'pliab(?:le|ility)'), ('flexible', 'flexib(?:le|ility)'), ('bumpy', 'bump(?:y|ier|iest|iness)'), ('grainy', 'grain(?:y|ier|iest|iness)'), ('ribbed', 'ribbed'), ('textured', 'textured'), ('abrasive', 'abrasive(?:ness)?'), ('slick', 'slick(?:ness|er|est)?'), ('slippery', 'slipper(?:y|ier|iest|iness)'), ('crisp', 'crisp(?:ness|er|est)?'), ('papery', 'paper(?:y|[- ]like)'), ('rubbery', 'rubber(?:y|[- ]like)'), ('leathery', 'leather(?:y|[- ]like)'), ('spongy', 'spong(?:y|ier|iest|iness)'), ('squishy', 'squish(?:y|ier|iest|iness)'), ('prickly', 'prickl(?:y|ier|iest|iness)'), ('nubby', 'nubb(?:y|ier|iest)'), ('drapey', 'drap(?:ey|y)'), ('lightweight', 'light[- ]?weight'), ('heavyweight', 'heavy[- ]?weight'), ('heavy', 'heav(?:y|ier|iest|iness)'), ('warm', 'warm(?:th|er|est)?'), ('cool', 'cool(?:ing|er|est|ness)?'), ('clammy', 'clamm(?:y|ier|iest|iness)'), ('sticky', 'stick(?:y|ier|iest|iness)'))
_CUE_NAMES = {f'cue{index}': term for index, (term, _) in enumerate(_CUES)}
_CUE_PATTERN = re.compile('(?<!\\w)(?:' + '|'.join((f'(?P<cue{index}>' + pattern + ')' for index, (_, pattern) in enumerate(_CUES))) + ')(?!\\w)', re.IGNORECASE)
_BOUNDARY = re.compile('(?<=[.!?;])(?=\\s|$)|[\\r\\n]+')
_CLAUSE_BOUNDARY = re.compile('[,;:]|\\b(?:but|however|although|yet)\\b', re.I)
_CONTEXT = re.compile('\\b(?:fabrics?|materials?|cloth|textiles?|garments?|apparel|clothing|shirts?|t[- ]?shirts?|tees?|blouses?|tops?|dress(?:es)?|skirts?|pants|trousers|jeans|leggings|shorts|sweaters?|cardigans?|jackets?|coats?|hoodies?|sweatshirts?|pullovers?|socks?|underwear|bras?|knits?|fleece|denim|cotton|linings?|surfaces?|seams?|waistbands?|sleeves?|collars?|cuffs?|textures?|skin|touch|feels?|felt|feeling)\\b', re.I)
_HYPOTHETICAL = re.compile('\\b(?:wish(?:ed|ing)?|if|would|could|should|will|can|may|might|maybe|perhaps|hope(?:d)?|hoping|expect(?:ed|ing)?|prefer(?:red)?|want(?:ed)?|suppos(?:ed|e))\\b|\\b(?:looking|searching)\\s+for\\b', re.I)
_COMPARISON = re.compile('\\b(?:than|compared|comparatively|relative\\s+to)\\b|\\b(?:more|less|as)\\s+(?:\\w+\\s+){0,2}(?:soft|rough|smooth|stiff|thin|thick|silky|scratchy|itchy|warm|cool|stretchy|coarse|plush|fuzzy|heavy|lightweight)\\b|\\b(?:softer|softest|rougher|roughest|smoother|smoothest|stiffer|stiffest|thinner|thinnest|thicker|thickest|silkier|silkiest|scratchier|scratchiest|itchier|itchiest|warmer|warmest|cooler|coolest|stretchier|stretchiest|coarser|coarsest|plusher|plushest|fuzzier|fuzziest|heavier|heaviest|harder|hardest|harshest|harsher|fluffier|fluffiest|bumpier|bumpiest|grainier|grainiest|slipperier|slipperiest|crisper|crispest)\\b', re.I)
_NEGATOR = "(?:not|never|no|neither|nor|isn['’]t|wasn['’]t|aren['’]t|weren['’]t|doesn['’]t|don['’]t|didn['’]t)"
_DIRECT_NEGATIVE = re.compile('\\b' + _NEGATOR + '\\s+(?:(?:feel|feels|felt|feeling|seem|seems|seemed)\\s+)?(?:at\\s+all\\s+)?$', re.I)
_ANY_NEGATIVE = re.compile('\\b' + _NEGATOR + '\\b', re.I)
_FALSE_NEGATIVE = re.compile('\\bnot\\s+(?:only|just)\\b', re.I)
_DEGREE_NEGATIVE = re.compile('\\b' + _NEGATOR + '\\s+(?:(?:feel|feels|felt|seem|seems|seemed)\\s+)?(?:too|very|really|particularly|especially|quite|that|so|as|sufficiently)\\b', re.I)
_NONPHYSICAL_AFTER: dict[str, str] = {'soft': '(?:colors?|colours?|shades?|pastels?|pink|blue|lighting|tones?|voice|sound|spot|heart|drinks?)\\b', 'smooth': '(?:transaction|delivery|shipping|process|return|experience|service|sale|ordering)\\b', 'rough': '(?:time|day|week|estimate|calculation|delivery|shipping|handling)\\b', 'hard': '(?:to|time|work|earned|working|water|decision|choice|pressed)\\b', 'stiff': '(?:price|cost|competition|penalty|fine)\\b', 'thin': '(?:wallet|patience|air|excuse|people|persons?|women|woman|men|man|girls?|boys?|body|frame|waist|hips?|legs?|thighs?|hair)\\b', 'thick': '(?:people|persons?|women|woman|men|man|body|waist|hips?|legs?|thighs?|hair)\\b', 'heavy': '(?:use|usage|duty|traffic|work|rain|snow|demand)\\b', 'warm': '(?:colors?|colours?|tones?|welcome|heart|regards|wishes)\\b', 'cool': '(?:design|look|style|colors?|colours?|pattern|print|idea|gift|brand)\\b', 'stretchy': '(?:goals?|assignment|budget|imagination)\\b', 'flexible': '(?:schedule|payment|return|policy|budget|hours)\\b', 'sticky': '(?:situation|price|label|sticker|tape)\\b'}

def _sentences(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for boundary in _BOUNDARY.finditer(text):
        end = boundary.start()
        if end > start:
            ranges.append((start, end))
        start = boundary.end()
    if start < len(text):
        ranges.append((start, len(text)))
    trimmed = []
    for start, end in ranges:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            trimmed.append((start, end))
    return trimmed

def _literal_cues(sentence: str) -> list[tuple[str, re.Match[str]]]:
    found = []
    for match in _CUE_PATTERN.finditer(sentence):
        term = _CUE_NAMES[match.lastgroup]
        following = sentence[match.end():].lstrip(' \t-,:"\'“”')
        reject = _NONPHYSICAL_AFTER.get(term)
        if reject and re.match(reject, following, re.I):
            continue
        if term in {'thin', 'thick', 'heavy'} and re.search("\\b(?:(?:i|he|she|they|we|you)\\s+(?:am|is|are|was|were)|i['’]m|he['’]s|she['’]s|they['’]re|we['’]re|you['’]re)\\s+(?:(?:quite|very|rather|too|a|little|extremely|pretty|fairly|not|so)\\s+){0,3}$", sentence[:match.start()], re.I):
            continue
        if term == 'cool' and (not re.search('\\b(?:skin|touch|temperature|heat|hot|summer|breathable|cooling)\\b|\\b(?:feels?|felt|keeps?|stays?|kept)\\s+(?:\\w+\\s+){0,3}cool\\b', sentence, re.I)):
            continue
        found.append((term, match))
    return found

def has_tactile_cue(text: str) -> bool:
    return bool(_CUE_PATTERN.search(text))

def extract_spans(text: str) -> list[dict[str, Any]]:
    if not isinstance(text, str):
        raise TypeError('text must be the original review string')
    spans = []
    for start, end in _sentences(text):
        sentence = text[start:end]
        terms = list(dict.fromkeys((term for term, _ in _literal_cues(sentence))))
        if terms:
            spans.append({'start': start, 'end': end, 'text': sentence, 'terms': terms})
    return spans

def _clause_for(sentence: str, term_start: int, term_end: int) -> tuple[str, int, int]:
    start, end = (0, len(sentence))
    for boundary in _CLAUSE_BOUNDARY.finditer(sentence):
        if boundary.end() <= term_start:
            start = boundary.end()
        elif boundary.start() >= term_end:
            end = boundary.start()
            break
    return (sentence[start:end], term_start - start, term_end - start)

def extract_evidence(text: str) -> list[dict[str, Any]]:
    evidence = []
    for span in extract_spans(text):
        sentence = span['text']
        if not _CONTEXT.search(sentence):
            continue
        if '?' in sentence or _HYPOTHETICAL.search(sentence):
            continue
        if _COMPARISON.search(sentence):
            continue
        for term, match in _literal_cues(sentence):
            clause, rel_start, rel_end = _clause_for(sentence, match.start(), match.end())
            prefix = clause[:rel_start]
            suffix = clause[rel_end:]
            if re.search('\\b(?:hardly|barely|scarcely|without)\\s*$', prefix, re.I):
                continue
            if re.search('\\b(?:without|unless)\\b', clause, re.I):
                continue
            if _DEGREE_NEGATIVE.search(prefix):
                continue
            if re.match('\\s+(?:it\\s+)?(?:is|was|are|were)\\s+not\\b', suffix, re.I):
                continue
            scope_prefix = _FALSE_NEGATIVE.sub('', prefix)
            negative = _DIRECT_NEGATIVE.search(scope_prefix)
            if negative:
                if re.search('\\benough\\b', suffix[:25], re.I):
                    continue
                if re.search('\\bnor\\b', negative.group()) and (not re.search('\\bneither\\b', sentence, re.I)):
                    continue
                polarity = -1
            elif _ANY_NEGATIVE.search(scope_prefix):
                continue
            else:
                polarity = 1
            evidence.append({'term': term, 'polarity': polarity, 'start': span['start'], 'end': span['end'], 'text': sentence, 'cue_start': span['start'] + match.start(), 'cue_end': span['start'] + match.end(), 'cue_text': match.group()})
    return evidence

def resolve_axis(authors, poles):
    positive = set()
    negative = set()
    support = {p: set() for p in poles}
    for author, records in authors.items():
        for term, sign in records:
            if term not in poles:
                continue
            if sign == 1:
                positive.add(term)
                support[term].add(author)
            elif sign == -1:
                negative.add(term)
            else:
                raise ValueError('Invalid explicit polarity')
    if negative:
        status = 'explicit_negative_excluded'
    elif len(positive) == 2:
        status = 'both_descriptors_excluded'
    elif len(positive) == 1:
        status = 'positive_descriptor' if poles[0] in positive else 'negative_descriptor'
    else:
        status = 'unknown'
    known = status in ('positive_descriptor', 'negative_descriptor')
    label = 1.0 if status == 'positive_descriptor' else 0.0 if status == 'negative_descriptor' else None
    return {'status': status, 'known': known, 'label': label, 'reported_descriptor': next(iter(positive)) if known else None, 'target_support': len(support[next(iter(positive))]) if known else 0, 'positive_pole_authors': len(support[poles[0]]), 'negative_pole_authors': len(support[poles[1]]), 'explicit_negative_descriptors': sorted(negative), 'all_target_authors': len(authors)}
