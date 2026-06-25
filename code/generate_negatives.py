#!/usr/bin/env python3
"""
generate_negatives.py

Generates synthetic negative (incorrectly translated) examples for MSLG-SPA pairs.

Usage:
    python generate_negatives.py [--input FILE] [--output FILE] [--n N] [--seed S] [--top-n N]
"""

import argparse
import json
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ─── Stop words ─────────────────────────────────────────────────────────────────

SPANISH_STOPWORDS = {
    'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con',
    'contra', 'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante',
    'e', 'el', 'ella', 'ellas', 'ellos', 'en', 'entre', 'era', 'eran',
    'eras', 'eres', 'es', 'esa', 'esas', 'ese', 'eso', 'esos', 'esta',
    'estaba', 'estaban', 'estado', 'estamos', 'están', 'estás', 'este',
    'estos', 'estoy', 'fue', 'fueron', 'fui', 'fuimos', 'ha', 'hace',
    'hacen', 'hacia', 'han', 'has', 'hasta', 'hay', 'he', 'la', 'las',
    'le', 'les', 'lo', 'los', 'me', 'mi', 'mía', 'mías', 'mío', 'míos',
    'mis', 'misma', 'mismas', 'mismo', 'mismos', 'muy', 'más', 'nos',
    'nosotras', 'nosotros', 'nuestra', 'nuestras', 'nuestro', 'nuestros',
    'o', 'os', 'para', 'pero', 'por', 'que', 'quien', 'quienes', 'se',
    'sea', 'sean', 'seas', 'ser', 'si', 'sin', 'sobre', 'son', 'soy',
    'su', 'sus', 'suya', 'suyas', 'suyo', 'suyos', 'también', 'te',
    'tengo', 'ti', 'tiene', 'tienen', 'tienes', 'todo', 'todos', 'tu',
    'tus', 'tú', 'un', 'una', 'unas', 'unos', 'usted', 'ustedes',
    'vosotras', 'vosotros', 'vuestra', 'vuestras', 'vuestro', 'vuestros',
    'y', 'ya', 'yo'
}

# In MSLG, pronouns and basic deixis are kept as stopwords for frequency tables
MSLG_STOPWORDS = {
    'YO', 'TÚ', 'ÉL', 'ELLA', 'NOSOTROS', 'USTEDES', 'ELLOS', 'ELLAS',
    'ESE', 'ESA', 'ESTE', 'ESTA', 'ESO', 'ESTO', 'MÍ'
}


# ─── Synonym dictionaries ────────────────────────────────────────────────────────
# Curated manually; extend as needed after inspecting frequency tables.

SPANISH_SYNONYMS: Dict[str, List[str]] = {
    'feliz': ['contento', 'alegre', 'dichoso'],
    'triste': ['melancólico', 'afligido', 'apenado'],
    'bonito': ['lindo', 'hermoso', 'bello'],
    'feo': ['horrible', 'desagradable', 'grotesco'],
    'grande': ['enorme', 'inmenso', 'gigante'],
    'pequeño': ['chico', 'diminuto', 'reducido'],
    'casa': ['hogar', 'vivienda', 'residencia'],
    'niño': ['chico', 'muchacho', 'infante'],
    'niña': ['chica', 'muchacha', 'infante'],
    'mujer': ['señora', 'dama', 'fémina'],
    'hombre': ['señor', 'caballero', 'varón'],
    'rápido': ['veloz', 'ligero', 'ágil'],
    'lento': ['despacio', 'pausado', 'moroso'],
    'bueno': ['excelente', 'magnífico', 'estupendo'],
    'malo': ['pésimo', 'terrible', 'nefasto'],
    'viejo': ['anciano', 'mayor', 'envejecido'],
    'joven': ['juvenil', 'mozo', 'adolescente'],
    'caro': ['costoso', 'oneroso', 'elevado'],
    'barato': ['económico', 'módico', 'accesible'],
    'llegar': ['arribar', 'venir', 'aparecer'],
    'ir': ['marcharse', 'partir', 'dirigirse'],
    'comer': ['ingerir', 'consumir', 'devorar'],
    'hablar': ['decir', 'expresar', 'comentar'],
    'ver': ['observar', 'mirar', 'contemplar'],
    'saber': ['conocer', 'entender', 'comprender'],
    'querer': ['desear', 'amar', 'anhelar'],
    'tener': ['poseer', 'contar', 'disponer'],
    'hacer': ['realizar', 'efectuar', 'ejecutar'],
    'lugar': ['sitio', 'espacio', 'zona'],
    'trabajo': ['empleo', 'labor', 'tarea'],
    'dinero': ['plata', 'capital', 'recursos'],
    'tiempo': ['época', 'período', 'momento'],
    'amigo': ['compañero', 'camarada', 'colega'],
    'familia': ['parientes', 'familiares', 'allegados'],
    'hijo': ['vástago', 'descendiente', 'retoño'],
    'hija': ['vástaga', 'descendiente', 'retoña'],
    'famoso': ['conocido', 'popular', 'célebre'],
    'alto': ['elevado', 'erguido', 'imponente'],
    'bajo': ['pequeño', 'corto', 'escaso'],
    'maestro': ['profesor', 'docente', 'instructor'],
    'maestra': ['profesora', 'docente', 'instructora'],
    'libro': ['texto', 'volumen', 'obra'],
    'agua': ['líquido', 'fluido', 'bebida'],
    'corona': ['diadema', 'tiara', 'tocado'],
    'juguete': ['juego', 'artefacto', 'entretenimiento'],
    'copia': ['duplicado', 'reproducción', 'fotocopia'],
    'campeonato': ['torneo', 'competencia', 'certamen'],
    'vecino': ['residente', 'habitante', 'colindante'],
    'constancia': ['certificado', 'documento', 'comprobante'],
    'desobedecer': ['incumplir', 'rebelarse', 'desacatar'],
    'expresivo': ['comunicativo', 'elocuente', 'locuaz'],
    'astuto': ['listo', 'sagaz', 'perspicaz'],
    'arrestado': ['detenido', 'capturado', 'apresado'],
    'ganar': ['obtener', 'lograr', 'conseguir'],
    'pagar': ['abonar', 'cancelar', 'liquidar'],
    'absorber': ['empapar', 'esponjar', 'succionar'],
    'gustar': ['agradar', 'encantar', 'fascinar'],
    'sacar': ['extraer', 'retirar', 'obtener'],
    'necesitar': ['requerir', 'precisar', 'demandar'],
    'usar': ['utilizar', 'emplear', 'ocupar'],
    'reunir': ['juntar', 'congregar', 'convocar'],
    'sobrino': ['pariente', 'familiar', 'allegado'],
    'persona': ['individuo', 'sujeto', 'ser'],
    'voluntaria': ['altruista', 'solidaria', 'colaboradora'],
    'antena': ['receptor', 'señal', 'dispositivo'],
}

MSLG_SYNONYMS: Dict[str, List[str]] = {
    'FELIZ': ['ALEGRE', 'CONTENTO'],
    'TRISTE': ['DEPRIMIDO', 'MELANCÓLICO'],
    'BONITO': ['LINDO', 'HERMOSO'],
    'FEO': ['HORRIBLE', 'DESAGRADABLE'],
    'GRANDE': ['ENORME', 'GIGANTE'],
    'PEQUEÑO': ['CHICO', 'DIMINUTO'],
    'CASA': ['HOGAR', 'VIVIENDA'],
    'RÁPIDO': ['VELOZ', 'LIGERO'],
    'BUENO': ['EXCELENTE', 'MAGNÍFICO'],
    'MALO': ['PÉSIMO', 'TERRIBLE'],
    'VIEJO': ['ANCIANO', 'MAYOR'],
    'JOVEN': ['JUVENIL', 'MOZO'],
    'CARO': ['COSTOSO', 'ELEVADO'],
    'BARATO': ['ECONÓMICO', 'MÓDICO'],
    'FAMOSO': ['CONOCIDO', 'POPULAR'],
    'ALTO': ['GRANDE', 'ELEVADO'],
    'BAJO': ['PEQUEÑO', 'CORTO'],
    'GUSTAR': ['AMAR', 'QUERER'],
    'NECESITAR': ['REQUERIR', 'PRECISAR'],
    'LLEGAR': ['ARRIBAR', 'VENIR'],
    'MAESTRO': ['PROFESOR', 'DOCENTE'],
    'LIBRO': ['TEXTO', 'VOLUMEN'],
    'AGUA': ['LÍQUIDO', 'BEBIDA'],
    'JUGUETE': ['JUEGO', 'ENTRETENIMIENTO'],
    'CAMPEONATO': ['TORNEO', 'COMPETENCIA'],
    'AYUDAR': ['APOYAR', 'ASISTIR'],
    'PAGAR': ['ABONAR', 'CANCELAR'],
    'GANAR': ['OBTENER', 'LOGRAR'],
    'USAR': ['UTILIZAR', 'EMPLEAR'],
    'COLOR': ['TONO', 'TONALIDAD'],
    'PERSONA': ['INDIVIDUO', 'SUJETO'],
    'LUGAR': ['SITIO', 'ESPACIO'],
    'CORONA': ['DIADEMA', 'TIARA'],
}


# ─── Tokenizers ─────────────────────────────────────────────────────────────────

def tokenize_mslg(text: str) -> List[str]:
    """Split MSLG gloss preserving compound (+), hyphenated, and dm-/# tokens."""
    return text.strip().split()


def tokenize_spa(text: str) -> List[str]:
    """Extract word tokens from Spanish text, stripping punctuation."""
    return re.findall(r'\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b', text, re.UNICODE)


def is_compound(token: str) -> bool:
    return '+' in token


def is_hyphenated_sign(token: str) -> bool:
    return '-' in token and not token.startswith('dm-')


def is_fingerspelled(token: str) -> bool:
    return token.startswith('#') or token.startswith('dm-')


# ─── Frequency tables ────────────────────────────────────────────────────────────

def build_freq_table(rows: List[str], tokenizer, stopwords: set) -> Counter:
    counter: Counter = Counter()
    for text in rows:
        for tok in tokenizer(text):
            if tok.lower() not in stopwords and tok.upper() not in stopwords and len(tok) > 1:
                counter[tok] += 1
    return counter


def freq_weighted_sample(freq_table: Counter, exclude: Optional[set] = None) -> Optional[str]:
    """Sample one token with probability proportional to its corpus frequency."""
    items = [(k, v) for k, v in freq_table.items()
             if exclude is None or k not in exclude]
    if not items:
        return None
    words, weights = zip(*items)
    return random.choices(words, weights=weights, k=1)[0]


# ─── Spanish morphological helpers ──────────────────────────────────────────────

_ARTICLE_GENDER = {
    'el': 'la', 'la': 'el', 'un': 'una', 'una': 'un',
    'los': 'las', 'las': 'los', 'unos': 'unas', 'unas': 'unos',
    'del': 'de la', 'al': 'a la',
}

_SPA_GENDER_SKIP = {'todo', 'poco', 'esto', 'eso', 'para', 'ya', 'era', 'esta', 'esa'}


def flip_gender_spa(word: str) -> Optional[str]:
    lw = word.lower()
    if lw in _ARTICLE_GENDER:
        r = _ARTICLE_GENDER[lw]
        return r.capitalize() if word[0].isupper() else r
    if lw.endswith('o') and len(lw) > 2 and lw not in _SPA_GENDER_SKIP:
        return word[:-1] + 'a'
    if lw.endswith('a') and len(lw) > 2 and lw not in _SPA_GENDER_SKIP:
        return word[:-1] + 'o'
    return None


def flip_plural_spa(word: str) -> Optional[str]:
    lw = word.lower()
    if lw.endswith('s') and len(lw) > 3:
        return word[:-1]            # singular: remove -s
    if lw[-1] in 'aeiouáéíóú':
        return word + 's'           # plural: add -s
    return word + 'es'              # plural: add -es


# ─── MSLG compound helpers ──────────────────────────────────────────────────────

def compound_swap(token: str) -> Optional[str]:
    """Swap the two parts of a compound sign: MAMÁ+PAPÁ → PAPÁ+MAMÁ."""
    parts = token.split('+')
    if len(parts) >= 2:
        parts[0], parts[1] = parts[1], parts[0]
        return '+'.join(parts)
    return None


def compound_split(token: str) -> List[str]:
    """Break MAMÁ+PAPÁ into ['MAMÁ', 'PAPÁ'] (two separate signs)."""
    return token.split('+')


# ─── Method C: Temporal / aspectual marker constants ────────────────────────────

MSLG_TEMPORAL_SWAPS: Dict[str, List[str]] = {
    'YA':      ['MAÑANA', 'ANTES', 'DESPUÉS'],
    'AYER':    ['MAÑANA', 'HOY', 'DESPUÉS'],
    'MAÑANA':  ['AYER', 'HOY', 'YA'],
    'HOY':     ['AYER', 'MAÑANA'],
    'SIEMPRE': ['NUNCA', 'A-VECES'],
    'NUNCA':   ['SIEMPRE', 'A-VECES'],
    'DESPUÉS': ['ANTES', 'AYER'],
    'ANTES':   ['DESPUÉS', 'MAÑANA'],
    'PRONTO':  ['DESPUÉS', 'TARDE'],
}

# Temporal adverb swaps (case-preserving via paired keys)
SPA_TEMPORAL_SWAPS: Dict[str, str] = {
    'ayer':    'mañana',  'Ayer':    'Mañana',
    'mañana':  'ayer',    'Mañana':  'Ayer',
    'hoy':     'ayer',    'Hoy':     'Ayer',
    'siempre': 'nunca',   'Siempre': 'Nunca',
    'nunca':   'siempre', 'Nunca':   'Siempre',
    'ya':      'todavía', 'Ya':      'Todavía',
    'todavía': 'ya',      'Todavía': 'Ya',
    'después': 'antes',   'Después': 'Antes',
    'antes':   'después', 'Antes':   'Después',
}

# Verb tense swaps: preterite ↔ future / present ↔ imperfect for common verbs
SPA_TENSE_SWAPS: Dict[str, str] = {
    # preterite 3sg ↔ future 3sg
    'llegó': 'llegará',   'llegará': 'llegó',
    'fue':   'irá',       'irá':     'fue',
    'compró':'comprará',  'comprará':'compró',
    'salió': 'saldrá',    'saldrá':  'salió',
    'vino':  'vendrá',    'vendrá':  'vino',
    'dijo':  'dirá',      'dirá':    'dijo',
    'hizo':  'hará',      'hará':    'hizo',
    'puso':  'pondrá',    'pondrá':  'puso',
    'tuvo':  'tendrá',    'tendrá':  'tuvo',
    'dio':   'dará',      'dará':    'dio',
    'vio':   'verá',      'verá':    'vio',
    'ganó':  'ganará',    'ganará':  'ganó',
    'pagó':  'pagará',    'pagará':  'pagó',
    'sacó':  'sacará',    'sacará':  'sacó',
    'usó':   'usará',     'usará':   'usó',
    'ayudó': 'ayudará',   'ayudará': 'ayudó',
    # preterite 1sg ↔ future 1sg
    'llegué':'llegaré',   'llegaré': 'llegué',
    'fui':   'iré',       'iré':     'fui',
    'compré':'compraré',  'compraré':'compré',
    'gané':  'ganaré',    'ganaré':  'gané',
    'pagué': 'pagaré',    'pagaré':  'pagué',
    'vi':    'veré',      'veré':    'vi',
    # present 3sg ↔ imperfect 3sg
    'está':  'estaba',    'estaba':  'está',
    'tiene': 'tenía',     'tenía':   'tiene',
    'gusta': 'gustaba',   'gustaba': 'gusta',
    'gustan':'gustaban',  'gustaban':'gustan',
    'quiere':'quería',    'quería':  'quiere',
    'puede': 'podía',     'podía':   'puede',
    'sabe':  'sabía',     'sabía':   'sabe',
    'hace':  'hacía',     'hacía':   'hace',
    'vive':  'vivía',     'vivía':   'vive',
    'debe':  'debía',     'debía':   'debe',
    'va':    'iba',       'iba':     'va',
}


# ─── Perturbation engine ─────────────────────────────────────────────────────────

def perturb_tokens(
    tokens: List[str],
    freq_table: Counter,
    synonyms: Dict[str, List[str]],
    is_mslg: bool,
    n_ops: int,
) -> Tuple[List[str], List[str]]:
    """
    Apply n_ops random perturbation operations.
    Returns (modified_tokens, operation_log).
    """
    tokens = list(tokens)
    log: List[str] = []

    base_ops = ['delete', 'insert', 'substitute', 'synonym', 'temporal_swap']
    extra_ops = (['compound_swap', 'compound_split'] if is_mslg
                 else ['gender_flip', 'plural_flip'])
    available_ops = base_ops + extra_ops

    def compound_indices() -> List[int]:
        return [i for i, t in enumerate(tokens) if is_compound(t)]

    for _ in range(n_ops):
        if not tokens:
            break
        op = random.choice(available_ops)
        idx = random.randint(0, len(tokens) - 1)

        # ── delete ────────────────────────────────
        if op == 'delete' and len(tokens) > 1:
            log.append(f"delete({tokens.pop(idx)})")

        # ── insert ────────────────────────────────
        elif op == 'insert':
            w = freq_weighted_sample(freq_table, exclude=set(tokens))
            if w:
                tokens.insert(idx, w)
                log.append(f"insert({w}@{idx})")

        # ── substitute ────────────────────────────
        elif op == 'substitute':
            w = freq_weighted_sample(freq_table, exclude={tokens[idx]})
            if w:
                log.append(f"sub({tokens[idx]}→{w})")
                tokens[idx] = w

        # ── synonym ───────────────────────────────
        elif op == 'synonym':
            key = tokens[idx] if is_mslg else tokens[idx].lower()
            if key in synonyms:
                replacement = random.choice(synonyms[key])
                log.append(f"syn({tokens[idx]}→{replacement})")
                tokens[idx] = replacement
            else:
                # no synonym available: fall back to freq-based substitution
                w = freq_weighted_sample(freq_table, exclude={tokens[idx]})
                if w:
                    log.append(f"sub({tokens[idx]}→{w})")
                    tokens[idx] = w

        # ── gender flip (SPA only) ─────────────────
        elif op == 'gender_flip':
            flipped = flip_gender_spa(tokens[idx])
            if flipped:
                log.append(f"gender({tokens[idx]}→{flipped})")
                tokens[idx] = flipped

        # ── plural flip (SPA only) ─────────────────
        elif op == 'plural_flip':
            flipped = flip_plural_spa(tokens[idx])
            if flipped:
                log.append(f"plural({tokens[idx]}→{flipped})")
                tokens[idx] = flipped

        # ── compound swap (MSLG only) ──────────────
        elif op == 'compound_swap':
            ci = compound_indices()
            if ci:
                i = random.choice(ci)
                swapped = compound_swap(tokens[i])
                if swapped:
                    log.append(f"cswap({tokens[i]}→{swapped})")
                    tokens[i] = swapped

        # ── compound split (MSLG only) ─────────────
        elif op == 'compound_split':
            ci = compound_indices()
            if ci:
                i = random.choice(ci)
                parts = compound_split(tokens[i])
                log.append(f"csplit({tokens[i]}→[{' '.join(parts)}])")
                tokens = tokens[:i] + parts + tokens[i + 1:]

        # ── temporal swap (both columns) ───────────
        elif op == 'temporal_swap':
            if is_mslg:
                t_idx = [i for i, t in enumerate(tokens) if t in MSLG_TEMPORAL_SWAPS]
                if t_idx:
                    i = random.choice(t_idx)
                    replacement = random.choice(MSLG_TEMPORAL_SWAPS[tokens[i]])
                    log.append(f"temporal({tokens[i]}→{replacement})")
                    tokens[i] = replacement
            else:
                # Check adverbs first, then verb tenses
                combined = {**SPA_TEMPORAL_SWAPS, **SPA_TENSE_SWAPS}
                t_idx = [(i, combined[t]) for i, t in enumerate(tokens) if t in combined]
                if t_idx:
                    i, replacement = random.choice(t_idx)
                    log.append(f"temporal({tokens[i]}→{replacement})")
                    tokens[i] = replacement

    return tokens, log


def reconstruct_spa(original: str, new_tokens: List[str]) -> str:
    """Re-assemble Spanish sentence preserving original terminal punctuation."""
    result = ' '.join(new_tokens)
    stripped = original.strip()
    if stripped.endswith('?'):
        result = '¿' + result + '?'
    elif stripped.endswith('.'):
        result += '.'
    elif stripped.endswith('!'):
        result = '¡' + result + '!'
    return result


# ─── Method A: Hard negatives via TF-IDF semantic pairing ───────────────────────

def generate_hard_negatives_tfidf(
    df: pd.DataFrame,
    n_samples: int,
    sim_min: float = 0.10,
    sim_max: float = 0.50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Find pairs of sentences with overlapping vocabulary (topically similar but
    distinct), then cross-pair MSLG from one row with SPA from the other.
    These hard negatives look plausible but are wrong translations.

    sim_min/sim_max define the TF-IDF cosine similarity window on the SPA column.
    Pairs below sim_min are too unrelated; above sim_max are near-duplicates.
    """
    if not _SKLEARN_AVAILABLE:
        print("WARNING: scikit-learn not installed — skipping hard negatives. "
              "Install with: pip install scikit-learn")
        return pd.DataFrame()

    random.seed(seed)
    np.random.seed(seed)

    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                                 min_df=1, strip_accents=None)
    tfidf = vectorizer.fit_transform(df['SPA'].tolist())
    sim = cosine_similarity(tfidf)          # (N, N) dense matrix

    n = len(df)
    # Collect all qualifying pairs (upper triangle only to avoid duplicates)
    candidates: List[Tuple[int, int, float]] = [
        (i, j, float(sim[i, j]))
        for i in range(n)
        for j in range(i + 1, n)
        if sim_min <= sim[i, j] <= sim_max
    ]
    random.shuffle(candidates)

    if not candidates:
        print(f"WARNING: No pairs found in similarity range [{sim_min}, {sim_max}]. "
              "Try widening --sim-min / --sim-max.")
        return pd.DataFrame()

    records = []
    for i, j, s in candidates:
        if len(records) >= n_samples:
            break
        # Two directions per pair
        for src, tgt in ((i, j), (j, i)):
            if len(records) >= n_samples:
                break
            row_s, row_t = df.iloc[src], df.iloc[tgt]
            records.append({
                'orig_id':       row_s['ID'],
                'MSLG':          row_s['MSLG'],
                'SPA':           row_t['SPA'],
                'perturbed_col': 'hard_negative',
                'operations':    (f"tfidf_pair(id_mslg={row_s['ID']},"
                                  f"id_spa={row_t['ID']},sim={s:.2f})"),
                'label': 0,
            })

    print(f"  {len(candidates)} candidate pairs in similarity range "
          f"[{sim_min}, {sim_max}]; generated {len(records)} hard negatives.")
    return pd.DataFrame(records)


# ─── Negative generation ─────────────────────────────────────────────────────────

def generate_negatives(
    df: pd.DataFrame,
    mslg_freq: Counter,
    spa_freq: Counter,
    n_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    records = []

    while len(records) < n_samples:
        row = df.sample(1).iloc[0]
        orig_mslg: str = row['MSLG']
        orig_spa: str = row['SPA']

        col = random.choice(['MSLG', 'SPA'])
        n_ops = random.randint(1, 3)

        if col == 'MSLG':
            tokens = tokenize_mslg(orig_mslg)
            new_tokens, log = perturb_tokens(tokens, mslg_freq, MSLG_SYNONYMS,
                                             is_mslg=True, n_ops=n_ops)
            new_mslg = ' '.join(new_tokens)
            new_spa = orig_spa
        else:
            tokens = tokenize_spa(orig_spa)
            new_tokens, log = perturb_tokens(tokens, spa_freq, SPANISH_SYNONYMS,
                                             is_mslg=False, n_ops=n_ops)
            new_mslg = orig_mslg
            new_spa = reconstruct_spa(orig_spa, new_tokens)

        # Skip if perturbation left the pair unchanged
        if new_mslg == orig_mslg and new_spa == orig_spa:
            continue

        records.append({
            'orig_id': row['ID'],
            'MSLG': new_mslg,
            'SPA': new_spa,
            'perturbed_col': col,
            'operations': '; '.join(log),
            'label': 0,
        })

    return pd.DataFrame(records)


# ─── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate synthetic negative MSLG-SPA translation pairs.')
    parser.add_argument('--input',  default='MSLG_SPA_train.txt',
                        help='Input TSV (ID, MSLG, SPA)')
    parser.add_argument('--output', default='negatives.csv',
                        help='Output CSV with positive + negative examples')
    parser.add_argument('--n',      type=int, default=500,
                        help='Number of negative examples to generate')
    parser.add_argument('--seed',   type=int, default=42)
    parser.add_argument('--top-n',   type=int, default=25,
                        dest='top_n',
                        help='Print top-N words from each frequency table')
    parser.add_argument('--n-hard',  type=int, default=100, dest='n_hard',
                        help='Hard negatives via TF-IDF pairing (method A; 0 to skip)')
    parser.add_argument('--sim-min', type=float, default=0.10, dest='sim_min',
                        help='Min TF-IDF cosine similarity for hard-negative pairs')
    parser.add_argument('--sim-max', type=float, default=0.50, dest='sim_max',
                        help='Max TF-IDF cosine similarity for hard-negative pairs')
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(args.input, sep='\t')
    print(f"Loaded {len(df)} records from '{args.input}'.\n")

    # ── Frequency tables ──────────────────────────────────────────────────────
    mslg_freq = build_freq_table(df['MSLG'].tolist(), tokenize_mslg, MSLG_STOPWORDS)
    spa_freq  = build_freq_table(df['SPA'].tolist(),  tokenize_spa,  SPANISH_STOPWORDS)

    print(f"Top {args.top_n} MSLG tokens (excluding stop words):")
    for word, cnt in mslg_freq.most_common(args.top_n):
        print(f"  {word:<35} {cnt}")

    print(f"\nTop {args.top_n} SPA tokens (excluding stop words):")
    for word, cnt in spa_freq.most_common(args.top_n):
        print(f"  {word:<35} {cnt}")

    with open('mslg_freq.json', 'w', encoding='utf-8') as f:
        json.dump(dict(mslg_freq.most_common()), f, ensure_ascii=False, indent=2)
    with open('spa_freq.json', 'w', encoding='utf-8') as f:
        json.dump(dict(spa_freq.most_common()), f, ensure_ascii=False, indent=2)
    print("\nFrequency tables saved → mslg_freq.json, spa_freq.json")

    # ── Generate perturbation-based negatives (includes temporal_swap as an op) ──
    print("\nGenerating perturbation-based negatives (methods: token ops + temporal)…")
    neg_df = generate_negatives(df, mslg_freq, spa_freq, args.n, args.seed)

    # ── Generate hard negatives via TF-IDF (method A) ─────────────────────────
    hard_df = pd.DataFrame()
    if args.n_hard > 0:
        print(f"\nGenerating {args.n_hard} TF-IDF hard negatives…")
        hard_df = generate_hard_negatives_tfidf(
            df, args.n_hard, args.sim_min, args.sim_max, args.seed)

    # ── Combine with positives ────────────────────────────────────────────────
    pos_df = df[['ID', 'MSLG', 'SPA']].rename(columns={'ID': 'orig_id'}).copy()
    pos_df['perturbed_col'] = ''
    pos_df['operations']    = ''
    pos_df['label']         = 1

    parts = [pos_df, neg_df]
    if not hard_df.empty:
        parts.append(hard_df)
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(args.output, index=False, encoding='utf-8')

    total_neg = len(neg_df) + len(hard_df)
    print(f"\nSaved {len(combined)} total examples  "
          f"({len(pos_df)} positive, {total_neg} negative  "
          f"[{len(neg_df)} perturbed + {len(hard_df)} hard])  →  '{args.output}'")

    # ── Show sample negatives ─────────────────────────────────────────────────
    print("\n── Sample perturbation negatives ─────────────────────────────────────")
    for _, row in neg_df.sample(min(4, len(neg_df)), random_state=args.seed).iterrows():
        print(f"  [{row['perturbed_col']}]  {row['operations']}")
        print(f"    MSLG : {row['MSLG']}")
        print(f"    SPA  : {row['SPA']}")
        print()

    if not hard_df.empty:
        print("── Sample hard negatives (TF-IDF) ────────────────────────────────────")
        for _, row in hard_df.sample(min(3, len(hard_df)), random_state=args.seed).iterrows():
            print(f"  {row['operations']}")
            print(f"    MSLG : {row['MSLG']}")
            print(f"    SPA  : {row['SPA']}")
            print()


if __name__ == '__main__':
    main()
