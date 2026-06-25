#!/usr/bin/env python3
"""
train_classifier_v2.py

SpaCy-enhanced binary classifier for MSLG-SPA translation pair quality.

Feature groups (54 total):
  V1 (21)  — length, MSLG structure, negation, temporal, similarity,
               + mslg_has_mujer_sign, mslg_has_mucho
  SpaCy SPA (27) — POS counts/ratios, gender, number,
                    tense binaries (pres/past/participio/cnd/ger/inf),
                    SPA presence flags (propn, fem-human noun, intensity word)
  Cross-modal (6) — temporal↔tense agree, name↔PER entity, content ratio,
                     + rule_mujer_fem, rule_mucho_intensity, rule_dm_propn

Label 1 = correct translation, label 0 = incorrect.

Usage:
    python train_classifier_v2.py --data negatives.csv [--cv 5] [--save-model model_v2.pkl]

Requirements:
    pip install spacy
    python -m spacy download es_core_news_md
"""

import argparse
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─── SpaCy (loaded once; parser/senter not needed) ───────────────────────────
try:
    import spacy
    NLP = spacy.load('es_core_news_md', disable=['parser', 'senter'])
except OSError:
    raise SystemExit(
        "SpaCy model 'es_core_news_md' not found.\n"
        "Install with:  python -m spacy download es_core_news_md"
    )

from sklearn.ensemble import (GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

N_SELECT = 20  # features to keep after pre-selection

# ─── Lexical constants ────────────────────────────────────────────────────────

PREFIX_LEN = 4

SPA_STOPWORDS = {
    'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con',
    'contra', 'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante',
    'e', 'el', 'ella', 'ellas', 'ellos', 'en', 'entre', 'era', 'eran',
    'eres', 'es', 'esa', 'ese', 'eso', 'esta', 'este', 'esto', 'estoy',
    'fue', 'fueron', 'fui', 'ha', 'hace', 'han', 'has', 'hasta', 'hay',
    'he', 'la', 'las', 'le', 'les', 'lo', 'los', 'me', 'mi', 'mis',
    'muy', 'más', 'nos', 'nosotros', 'o', 'os', 'para', 'pero', 'por',
    'que', 'quien', 'se', 'sea', 'ser', 'si', 'sin', 'sobre', 'son',
    'soy', 'su', 'sus', 'también', 'te', 'ti', 'tiene', 'tienen', 'todo',
    'todos', 'tu', 'tus', 'tú', 'un', 'una', 'unas', 'unos', 'usted',
    'ustedes', 'y', 'yo',
}

MSLG_STOPWORDS = {
    'YO', 'TÚ', 'ÉL', 'ELLA', 'NOSOTROS', 'USTEDES', 'ELLOS', 'ELLAS',
    'ESE', 'ESA', 'ESTE', 'ESTA', 'ESO', 'ESTO', 'MÍ',
}
#SPA_STOPWORDS = {}
#MSLG_STOPWORDS = {}

MSLG_TEMPORALS = {'YA', 'AYER', 'MAÑANA', 'HOY', 'SIEMPRE', 'NUNCA',
                  'DESPUÉS', 'ANTES', 'PRONTO', 'TARDE'}

SPA_TEMPORALS = {'ayer', 'mañana', 'hoy', 'siempre', 'nunca', 'ya',
                 'todavía', 'después', 'antes', 'pronto'}

MSLG_INTERROGATIVES = {'QUÉ', 'QUIÉN', 'CÓMO', 'CUÁNDO', 'DÓNDE',
                        'POR-QUÉ', 'CUÁNTO', 'CUÁL'}

SPA_NEGATIONS = {'no', 'nunca', 'jamás', 'tampoco', 'ningún', 'ninguna'}

# MSLG temporal → expected SPA verb tenses (spaCy Tense morph value)
MSLG_TEMP_TO_TENSE: Dict[str, set] = {
    'AYER':    {'Past'},
    'MAÑANA':  {'Fut', 'Pres'},     # SPA uses present-for-future frequently
    'YA':      {'Past', 'Pres'},
    'HOY':     {'Pres'},
    'ANTES':   {'Past', 'Pres'},
    'DESPUÉS': {'Fut', 'Pres'},
    'SIEMPRE': {'Pres', 'Past'},
    'NUNCA':   {'Pres', 'Past'},
    'PRONTO':  {'Fut', 'Pres'},
    'TARDE':   {'Pres', 'Past'},
}

# Feminine human nouns in SPA that correspond to X+MUJER compounds in MSLG
# (e.g. HERMANO+MUJER = hermana, AMIGO+MUJER = amiga, HIJO+MUJER = hija)
SPA_FEM_HUMAN_NOUNS = {
    'hermana', 'amiga', 'novia', 'hija', 'tía', 'abuela', 'abuelita',
    'maestra', 'vecina', 'niña', 'señora', 'señorita', 'mujer',
    'madre', 'mamá', 'doctora', 'enfermera', 'directora', 'prima',
    'alumna', 'chica', 'esposa', 'jefa', 'presidenta', 'socia',
}

# Intensity / degree adverbs and quantifiers in SPA that map to MSLG MUCHO
SPA_INTENSITY_WORDS = {
    'muy', 'mucho', 'mucha', 'muchos', 'muchas',
    'demasiado', 'demasiada', 'demasiados', 'demasiadas',
    'bastante', 'tan', 'tanto', 'tanta', 'tantos', 'tantas',
}

# MSLG quantifier/intensifier tokens (correspond to SPA intensity words)
MSLG_MUCHO_TOKENS = {'MUCHO', 'MUCHOS', 'MUCHA', 'MUCHAS',
                     'TANTO', 'TANTA', 'TANTOS', 'TANTAS'}


# ─── Tokenizers & MSLG helpers ───────────────────────────────────────────────

def tokenize_mslg(text: str) -> List[str]:
    return text.strip().split()


def tokenize_spa(text: str) -> List[str]:
    return re.findall(r'\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b', text, re.UNICODE)


def is_fingerspelled(token: str) -> bool:
    return token.startswith('dm-') or token.startswith('#')


def normalize_mslg_token(token: str) -> str:
    if token.startswith('dm-'):
        return token[3:].lower()
    if token.startswith('#'):
        return token[1:].lower()
    if '+' in token:
        return ' '.join(p.lower() for p in token.split('+'))
    return token.lower()


# ─── Leakage-free prefix similarity ─────────────────────────────────────────

def _mslg_prefix_set(mslg: str) -> set:
    result = set()
    for tok in tokenize_mslg(mslg):
        if tok.upper() in MSLG_STOPWORDS:
            continue
        norm = normalize_mslg_token(tok)
        for part in norm.split():
            if len(part) >= 3:
                result.add(part[:PREFIX_LEN])
    return result


def _spa_prefix_set(spa: str) -> set:
    result = set()
    for tok in tokenize_spa(spa):
        lw = tok.lower()
        if lw not in SPA_STOPWORDS and len(lw) >= 3:
            result.add(lw[:PREFIX_LEN])
    return result


def prefix_overlap(mslg: str, spa: str) -> float:
    ms = _mslg_prefix_set(mslg)
    ss = _spa_prefix_set(spa)
    return len(ms & ss) / len(ms) if ms else 0.0


def jaccard_sim(mslg: str, spa: str) -> float:
    ms = _mslg_prefix_set(mslg)
    ss = _spa_prefix_set(spa)
    union = ms | ss
    return len(ms & ss) / len(union) if union else 1.0


# ─── Group 1: V1 features  (21, no spaCy) ───────────────────────────────────

V1_FEATURE_NAMES = [
    # Length
    'len_mslg', 'len_spa', 'len_ratio', 'len_diff_abs',
    # MSLG structure
    'n_compounds', 'n_hyphenated', 'n_fingerspelled',
    # Negation
    'has_neg_mslg', 'has_neg_spa', 'neg_match',
    # Temporal
    'has_temp_mslg', 'has_temp_spa', 'temp_match',
    # Interrogative
    'has_interrog_mslg', 'has_question_spa', 'interrog_match',
    # Fingerspelled name match
    'fs_name_match',
    # Leakage-free cross-modal similarity
    'prefix_overlap', 'jaccard_sim',
    # MSLG presence flags (new)
    'mslg_has_mujer_sign',   # any token is or contains +MUJER
    'mslg_has_mucho',        # MUCHO / MUCHOS / MUCHA / MUCHAS / TANTO etc.
]


def _extract_v1(mslg: str, spa: str) -> np.ndarray:
    mslg_toks = tokenize_mslg(mslg)
    spa_toks   = tokenize_spa(spa)

    len_m        = len(mslg_toks)
    len_s        = len(spa_toks)
    len_ratio    = len_m / (len_s + 1e-6)
    len_diff_abs = abs(len_m - len_s)

    n_compounds   = sum(1 for t in mslg_toks if '+' in t)
    n_hyphenated  = sum(1 for t in mslg_toks if '-' in t and not t.startswith('dm-'))
    n_fingerspell = sum(1 for t in mslg_toks if is_fingerspelled(t))

    has_neg_m = int('NO' in mslg_toks)
    has_neg_s = int(any(t.lower() in SPA_NEGATIONS for t in spa_toks))
    neg_match = int(has_neg_m == has_neg_s)

    has_temp_m = int(any(t in MSLG_TEMPORALS for t in mslg_toks))
    has_temp_s = int(any(t.lower() in SPA_TEMPORALS for t in spa_toks))
    temp_match = int(has_temp_m == has_temp_s)

    has_interrog_m = int(any(t in MSLG_INTERROGATIVES for t in mslg_toks)
                         or mslg.startswith('¿'))
    has_question_s = int('?' in spa)
    interrog_match = int(has_interrog_m == has_question_s)

    fs_names = {normalize_mslg_token(t).lower()
                for t in mslg_toks if is_fingerspelled(t)}
    if fs_names:
        spa_joined = ' '.join(spa_toks).lower()
        fs_name_match = int(all(name in spa_joined for name in fs_names))
    else:
        fs_name_match = 1

    p_overlap = prefix_overlap(mslg, spa)
    j_sim     = jaccard_sim(mslg, spa)

    # MSLG presence flags ──────────────────────────────────────────────────────
    # +MUJER compounds (HERMANO+MUJER=hermana, HIJO+MUJER=hija, etc.)
    # Excludes MUJERIEGO (a different word)
    mslg_has_mujer = int(
        any(t == 'MUJER' or '+MUJER' in t for t in mslg_toks)
    )
    mslg_has_mucho = int(any(t in MSLG_MUCHO_TOKENS for t in mslg_toks))

    return np.array([
        len_m, len_s, len_ratio, len_diff_abs,
        n_compounds, n_hyphenated, n_fingerspell,
        has_neg_m, has_neg_s, neg_match,
        has_temp_m, has_temp_s, temp_match,
        has_interrog_m, has_question_s, interrog_match,
        fs_name_match,
        p_overlap, j_sim,
        mslg_has_mujer, mslg_has_mucho,
    ], dtype=float)


# ─── Group 2: SpaCy SPA features  (27) ──────────────────────────────────────
#
# Verb tense detection reliability in es_core_news_md:
#   Reliable   → Tense=Past+Mood=Ind (indefinido), Tense=Fut, Mood=Cnd,
#                VerbForm=Part (participio), VerbForm=Ger, VerbForm=Inf
#   Unreliable → imperfecto (mis-tagged as subjuntivo pres), imperativo
#   Strategy   → use binary has_* flags, not raw counts for tense

SPACY_FEATURE_NAMES = [
    # POS counts and ratios
    'spa_n_verbs', 'spa_n_nouns', 'spa_n_adjs', 'spa_n_advs',
    'spa_verb_ratio', 'spa_noun_ratio',
    # Gender (tokens carrying Gender morph feature)
    'spa_n_masc', 'spa_n_fem', 'spa_gender_balance',
    # Number
    'spa_n_sing', 'spa_n_plur', 'spa_plural_ratio',
    # Tense counts (coarse — supplement with binaries below)
    'spa_n_pres', 'spa_n_past', 'spa_n_fut',
    # Tense binary flags (more reliable signal per token)
    'spa_has_pres_ind',    # Tense=Pres + Mood=Ind + VerbForm=Fin
    'spa_has_past_ind',    # Tense=Past + Mood=Ind + VerbForm=Fin  (indefinido)
    'spa_has_participio',  # VerbForm=Part + Tense=Past  (ha comido, había llegado)
    'spa_has_cnd',         # Mood=Cnd  (comería)
    'spa_has_ger',         # VerbForm=Ger  (comiendo, estando)
    'spa_n_inf',           # count of VerbForm=Inf (modal + nominal structures)
    'spa_has_subj',        # Mood=Sub  (coma, llegara) — noisy but informative
    # Named entities
    'spa_has_per_entity', 'spa_n_entities',
    # Presence flags used by cross-modal rules
    'spa_has_propn',       # any PROPN token
    'spa_has_fem_human',   # token lemma in SPA_FEM_HUMAN_NOUNS
    'spa_has_intensity',   # muy / mucho / bastante / tan etc.
]


def _extract_spacy(doc) -> np.ndarray:
    n_tok = max(len(doc), 1)
    is_verb = lambda t: t.pos_ in ('VERB', 'AUX')

    # POS
    n_verbs = sum(1 for t in doc if is_verb(t))
    n_nouns = sum(1 for t in doc if t.pos_ in ('NOUN', 'PROPN'))
    n_adjs  = sum(1 for t in doc if t.pos_ == 'ADJ')
    n_advs  = sum(1 for t in doc if t.pos_ == 'ADV')
    verb_ratio = n_verbs / n_tok
    noun_ratio = n_nouns / n_tok

    # Gender
    n_masc = sum(1 for t in doc if 'Masc' in t.morph.get('Gender'))
    n_fem  = sum(1 for t in doc if 'Fem'  in t.morph.get('Gender'))
    g_tot  = n_masc + n_fem
    gender_balance = n_masc / g_tot if g_tot > 0 else 0.5

    # Number
    n_sing = sum(1 for t in doc if 'Sing' in t.morph.get('Number'))
    n_plur = sum(1 for t in doc if 'Plur' in t.morph.get('Number'))
    num_tot = n_sing + n_plur
    plural_ratio = n_plur / num_tot if num_tot > 0 else 0.0

    # Tense counts (coarse)
    n_pres = sum(1 for t in doc if is_verb(t) and 'Pres' in t.morph.get('Tense'))
    n_past = sum(1 for t in doc if is_verb(t) and 'Past' in t.morph.get('Tense'))
    n_fut  = sum(1 for t in doc if is_verb(t) and 'Fut'  in t.morph.get('Tense'))

    # Tense binary flags (reliable signals)
    has_pres_ind  = int(any(
        is_verb(t)
        and 'Pres' in t.morph.get('Tense')
        and 'Ind'  in t.morph.get('Mood')
        and 'Fin'  in t.morph.get('VerbForm')
        for t in doc
    ))
    has_past_ind  = int(any(
        is_verb(t)
        and 'Past' in t.morph.get('Tense')
        and 'Ind'  in t.morph.get('Mood')
        and 'Fin'  in t.morph.get('VerbForm')
        for t in doc
    ))
    has_participio = int(any(
        'Part' in t.morph.get('VerbForm') and 'Past' in t.morph.get('Tense')
        for t in doc
    ))
    has_cnd  = int(any('Cnd' in t.morph.get('Mood') for t in doc if is_verb(t)))
    has_ger  = int(any('Ger' in t.morph.get('VerbForm') for t in doc))
    n_inf    = sum(1 for t in doc if 'Inf' in t.morph.get('VerbForm'))
    has_subj = int(any('Sub' in t.morph.get('Mood') for t in doc if is_verb(t)))

    # Named entities
    has_per    = int(any(ent.label_ == 'PER' for ent in doc.ents))
    n_entities = len(doc.ents)

    # Presence flags for cross-modal rules
    has_propn     = int(any(t.pos_ == 'PROPN' for t in doc))
    has_fem_human = int(any(t.lemma_.lower() in SPA_FEM_HUMAN_NOUNS for t in doc))
    has_intensity = int(any(t.text.lower() in SPA_INTENSITY_WORDS for t in doc))

    return np.array([
        n_verbs, n_nouns, n_adjs, n_advs,
        verb_ratio, noun_ratio,
        n_masc, n_fem, gender_balance,
        n_sing, n_plur, plural_ratio,
        n_pres, n_past, n_fut,
        has_pres_ind, has_past_ind, has_participio,
        has_cnd, has_ger, n_inf, has_subj,
        has_per, n_entities,
        has_propn, has_fem_human, has_intensity,
    ], dtype=float)


# ─── Group 3: Cross-modal features  (6) ─────────────────────────────────────
#
# Rule encoding convention:
#   1.0 → rule satisfied OR the MSLG trigger is absent (neutral)
#   0.0 → MSLG trigger present but SPA expected element is missing (violation)
#   0.5 → SPA trigger present but MSLG expected structure absent (soft violation)
#
# Each rule exposes the agreement signal without duplicating presence flags
# that already appear in V1 / SpaCy groups.

CROSSMODAL_FEATURE_NAMES = [
    'mslg_temp_tense_agree',     # MSLG temporal marker ↔ SPA verb tense
    'mslg_name_spa_per_agree',   # MSLG dm- name ↔ SPA PER entity / text
    'mslg_len_vs_spa_content_ratio',
    'rule_mujer_fem_agree',      # MSLG +MUJER ↔ SPA feminine human noun
    'rule_mucho_intensity_agree',# MSLG MUCHO ↔ SPA muy/mucho/bastante
    'rule_dm_propn_agree',       # MSLG dm- token ↔ SPA PROPN
]


def _agreement(trigger_a: bool, trigger_b: bool) -> float:
    """Symmetric agreement: 1 if both present or both absent, 0.5 if only one side."""
    if trigger_a and trigger_b:
        return 1.0
    if not trigger_a and not trigger_b:
        return 1.0
    return 0.5   # asymmetric presence — partial penalty; model learns the weight


def _extract_crossmodal(mslg: str, doc) -> np.ndarray:
    mslg_toks = tokenize_mslg(mslg)
    is_verb   = lambda t: t.pos_ in ('VERB', 'AUX')

    # ── Temporal ↔ verb tense ─────────────────────────────────────────────────
    temp_tokens = [t for t in mslg_toks if t in MSLG_TEMP_TO_TENSE]
    if not temp_tokens:
        temp_tense_agree = 1.0
    else:
        expected = set()
        for t in temp_tokens:
            expected |= MSLG_TEMP_TO_TENSE[t]
        actual = {tense
                  for tok in doc if is_verb(tok)
                  for tense in tok.morph.get('Tense')}
        if not actual:
            temp_tense_agree = 0.5      # no finite verbs found (nominal SPA)
        elif actual & expected:
            temp_tense_agree = 1.0
        else:
            temp_tense_agree = 0.0

    # ── MSLG dm- name ↔ SPA PER entity / substring ────────────────────────────
    dm_names = {normalize_mslg_token(t) for t in mslg_toks if t.startswith('dm-')}
    if not dm_names:
        name_per_agree = 1.0
    else:
        per_texts  = {ent.text.lower() for ent in doc.ents if ent.label_ == 'PER'}
        spa_lower  = doc.text.lower()
        found = all(
            name in spa_lower or any(name in pe for pe in per_texts)
            for name in dm_names
        )
        name_per_agree = float(found)

    # ── MSLG content tokens vs SPA content words ratio ────────────────────────
    n_mslg_content = sum(1 for t in mslg_toks if t not in MSLG_STOPWORDS)
    n_spa_content  = sum(1 for t in doc
                         if t.pos_ in ('NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ'))
    content_ratio  = n_mslg_content / max(n_spa_content, 1)

    # ── Rule: MSLG +MUJER  ↔  SPA feminine human noun ────────────────────────
    # Corpus pattern: HERMANO+MUJER=hermana, HIJO+MUJER=hija, AMIGO+MUJER=amiga
    mslg_has_mujer = any(t == 'MUJER' or '+MUJER' in t for t in mslg_toks)
    spa_has_fem    = any(t.lemma_.lower() in SPA_FEM_HUMAN_NOUNS for t in doc)
    rule_mujer     = _agreement(mslg_has_mujer, spa_has_fem)

    # ── Rule: MSLG MUCHO  ↔  SPA muy / mucho / bastante / tan ───────────────
    mslg_has_mucho   = any(t in MSLG_MUCHO_TOKENS for t in mslg_toks)
    spa_has_intensity = any(t.text.lower() in SPA_INTENSITY_WORDS for t in doc)
    rule_mucho        = _agreement(mslg_has_mucho, spa_has_intensity)

    # ── Rule: MSLG dm-  ↔  SPA PROPN (proper noun) ───────────────────────────
    mslg_has_dm  = any(t.startswith('dm-') for t in mslg_toks)
    spa_has_propn = any(t.pos_ == 'PROPN' for t in doc)
    rule_dm       = _agreement(mslg_has_dm, spa_has_propn)

    return np.array([
        temp_tense_agree, name_per_agree, content_ratio,
        rule_mujer, rule_mucho, rule_dm,
    ], dtype=float)


# ─── Full feature set ─────────────────────────────────────────────────────────

FEATURE_NAMES = V1_FEATURE_NAMES + SPACY_FEATURE_NAMES + CROSSMODAL_FEATURE_NAMES


def extract_row_features(mslg: str, spa: str, doc=None) -> np.ndarray:
    if doc is None:
        doc = NLP(spa)
    return np.concatenate([
        _extract_v1(mslg, spa),
        _extract_spacy(doc),
        _extract_crossmodal(mslg, doc),
    ])


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    docs = list(NLP.pipe(df['SPA'].tolist(), batch_size=64))
    return np.vstack([
        extract_row_features(row.MSLG, row.SPA, doc)
        for row, doc in zip(df.itertuples(index=False), docs)
    ])


# ─── Feature pre-selection ───────────────────────────────────────────────────

def preselect_features(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Select the top N_SELECT features by mutual information (fit on full dataset).
    Returns (X_reduced, selected_indices, selected_names).

    Mutual information is computed on unscaled features; the column indices
    are then stored in the model bundle so inference can apply the same mask.
    """
    sel = SelectKBest(
        lambda X, y: mutual_info_classif(X, y, random_state=42),
        k=N_SELECT,
    )
    sel.fit(X, y)
    indices = sel.get_support(indices=True).tolist()
    names   = [FEATURE_NAMES[i] for i in indices]

    print(f"\nPre-selected {N_SELECT} features (by mutual information):")
    scores = sel.scores_
    for rank, i in enumerate(indices, 1):
        print(f"  {rank:>2}. {FEATURE_NAMES[i]:<32}  MI={scores[i]:.4f}")

    return X[:, indices], indices, names


# ─── Models ──────────────────────────────────────────────────────────────────

def get_models() -> dict:
    return {
        'LogisticRegression': LogisticRegression(C=1.0, max_iter=1000,
                                                  class_weight='balanced'),
        'SVM_RBF':            SVC(kernel='rbf', C=1.0, gamma='scale',
                                  probability=True, class_weight='balanced'),
        'RandomForest':       RandomForestClassifier(n_estimators=150,
                                                     max_depth=3,
                                                     class_weight='balanced',
                                                     random_state=42),
        'GradientBoosting':   GradientBoostingClassifier(n_estimators=150,
                                                          max_depth=3,
                                                          learning_rate=0.025,
                                                          subsample=0.7,
                                                          random_state=42),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=150,
                                                                max_depth=3,
                                                                learning_rate=0.03,
                                                                min_samples_leaf=10,
                                                                class_weight='balanced',
                                                                random_state=42),
    }


# ─── Cross-validated evaluation ──────────────────────────────────────────────

def evaluate_cv(X_all: np.ndarray, y: np.ndarray,
                n_splits: int = 5) -> Tuple[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models    = get_models()
    results   = {n: {'roc_auc': [], 'pr_auc': []} for n in models}
    oof_preds = {n: np.zeros(len(y)) for n in models}

    print(f"\n{'─'*64}")
    print(f"Stratified {n_splits}-Fold CV  "
          f"({len(y)} examples, {int(y.sum())} pos / {int((~y.astype(bool)).sum())} neg)")
    print(f"Feature matrix: {X_all.shape[0]} × {X_all.shape[1]}  "
          f"(pre-selected from {len(FEATURE_NAMES)} total)")
    print(f"{'─'*64}\n")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y), 1):
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y[train_idx],     y[test_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        for name, model in models.items():
            model.fit(X_tr_sc, y_tr)
            proba = model.predict_proba(X_te_sc)[:, 1]
            oof_preds[name][test_idx] = proba
            results[name]['roc_auc'].append(roc_auc_score(y_te, proba))
            results[name]['pr_auc'].append(average_precision_score(y_te, proba))

        print(f"  Fold {fold_idx}  |  "
              + "  ".join(f"{n}: ROC={results[n]['roc_auc'][-1]:.3f}"
                          for n in models))

    print(f"\n{'─'*64}")
    print(f"{'Model':<22}  {'ROC-AUC':>10}  {'PR-AUC':>10}")
    print(f"{'─'*64}")
    best_name, best_roc = None, -1.0
    for name in models:
        roc_m = np.mean(results[name]['roc_auc'])
        roc_s = np.std( results[name]['roc_auc'])
        pr_m  = np.mean(results[name]['pr_auc'])
        pr_s  = np.std( results[name]['pr_auc'])
        marker = '  ←' if roc_m > best_roc else ''
        print(f"  {name:<20}  {roc_m:.3f}±{roc_s:.3f}  {pr_m:.3f}±{pr_s:.3f}{marker}")
        if roc_m > best_roc:
            best_roc = roc_m
            best_name = name
    print(f"{'─'*64}")

    print(f"\nBest model: {best_name}  (ROC-AUC {best_roc:.3f})")
    print("\nOut-of-fold classification report (threshold 0.5):")
    best_oof = oof_preds[best_name]
    preds = (best_oof >= 0.5).astype(int)
    print(classification_report(y, preds, target_names=['incorrect', 'correct'],
                                 digits=3))
    cm = confusion_matrix(y, preds)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"             pred_neg  pred_pos")
    print(f"  true_neg    {cm[0,0]:>6}    {cm[0,1]:>6}")
    print(f"  true_pos    {cm[1,0]:>6}    {cm[1,1]:>6}")

    return best_name, best_roc


# ─── Feature importance ───────────────────────────────────────────────────────

def show_feature_importance(X_all: np.ndarray, y: np.ndarray,
                             best_model_name: str,
                             feature_names: List[str] = None) -> None:
    names = feature_names if feature_names is not None else FEATURE_NAMES
    print(f"\n{'─'*64}")
    print(f"Feature importance — {best_model_name} (fit on full dataset, {len(names)} features)")
    print(f"{'─'*64}")

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_all)
    model = get_models()[best_model_name]
    model.fit(X_sc, y)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("  (model does not expose feature importances)")
        return

    order   = np.argsort(importances)[::-1]
    max_imp = importances[order[0]]

    v1_end    = len(V1_FEATURE_NAMES)
    spacy_end = v1_end + len(SPACY_FEATURE_NAMES)

    # Map selected-feature positions back to original group boundaries
    all_names = FEATURE_NAMES
    for rank, i in enumerate(order, 1):
        fname = names[i]
        orig  = all_names.index(fname) if fname in all_names else -1
        group = ('V1   ' if orig < v1_end   else
                 'spaCy' if orig < spacy_end else
                 'cross')
        bar = '█' * int(importances[i] / max_imp * 20)
        print(f"  {rank:>2}. [{group}] {fname:<32}  {importances[i]:.4f}  {bar}")


# ─── Save model bundle ────────────────────────────────────────────────────────

def save_model(X_all: np.ndarray, y: np.ndarray,
               best_model_name: str, path: str,
               sel_indices: List[int] = None) -> None:
    import pickle

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_all)
    model = get_models()[best_model_name]
    model.fit(X_sc, y)

    sel = sel_indices if sel_indices is not None else list(range(len(FEATURE_NAMES)))
    bundle = {
        'model':         model,
        'scaler':        scaler,
        'model_name':    best_model_name,
        'feature_names': [FEATURE_NAMES[i] for i in sel],
        'sel_indices':   sel,
        'spacy_model':   'es_core_news_md',
    }
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\nModel bundle saved → {path}")
    print("  Predict:  predict(bundle, mslg_text, spa_text)")


# ─── Inference ───────────────────────────────────────────────────────────────

def predict(bundle: dict, mslg: str, spa: str) -> Tuple[int, float]:
    """Return (label, p_correct). label 1 = likely correct, 0 = likely incorrect."""
    doc   = NLP(spa)
    feats = extract_row_features(mslg, spa, doc)
    X_sel = feats.reshape(1, -1)[:, bundle['sel_indices']]
    X_sc  = bundle['scaler'].transform(X_sel)
    prob  = bundle['model'].predict_proba(X_sc)[0, 1]
    return int(prob >= 0.5), float(prob)


def predict_csv(model_path: str, input_csv: str, output_csv: str) -> None:
    """
    Score every row in input_csv with a saved model bundle.

    input_csv must contain columns 'MSLG' and 'SPA'.
    Any additional columns are preserved in the output.

    Output columns added:
      p_correct  — P(correct translation), float in [0, 1]
      label_pred — predicted label at threshold 0.5  (1=correct, 0=incorrect)
    """
    import pickle

    print(f"Loading model from '{model_path}' ...")
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    print(f"  Model: {bundle['model_name']}  |  "
          f"Features: {len(bundle['feature_names'])}")

    df = pd.read_csv(input_csv)
    missing = [c for c in ('MSLG', 'SPA') if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    print(f"Loaded {len(df)} rows from '{input_csv}'")

    print(f"Building features ({len(FEATURE_NAMES)} per row) ...")
    X     = build_feature_matrix(df)
    X_sel = X[:, bundle['sel_indices']]
    X_sc  = bundle['scaler'].transform(X_sel)

    proba  = bundle['model'].predict_proba(X_sc)[:, 1]
    labels = (proba >= 0.5).astype(int)

    out = df.copy()
    out['p_correct']  = proba.round(4)
    out['label_pred'] = labels
    out.to_csv(output_csv, index=False, encoding='utf-8')

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    print(f"\nResults saved → '{output_csv}'")
    print(f"  Predicted correct:   {n_pos:>5}  ({100 * n_pos / len(labels):.1f}%)")
    print(f"  Predicted incorrect: {n_neg:>5}  ({100 * n_neg / len(labels):.1f}%)")
    print(f"  Mean p_correct:      {proba.mean():.3f}  (std {proba.std():.3f})")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import os

    parser = argparse.ArgumentParser(
        description='SpaCy-enhanced MSLG-SPA pair quality classifier (v2).',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Training arguments ────────────────────────────────────────────────────
    train_grp = parser.add_argument_group('Training')
    train_grp.add_argument('--data',       default='negatives.csv',
                           help='Labelled CSV from generate_negatives.py  (default: negatives.csv)')
    train_grp.add_argument('--cv',         type=int, default=5,
                           help='Number of CV folds  (default: 5)')
    train_grp.add_argument('--save-model', default='model_v2.pkl', dest='save_model',
                           help='Path to save the trained model bundle  (default: model_v2.pkl)')
    train_grp.add_argument('--no-save',    action='store_true', dest='no_save',
                           help='Skip saving the model after training')

    # ── Prediction arguments ──────────────────────────────────────────────────
    pred_grp = parser.add_argument_group('Prediction  (skips training when --predict is given)')
    pred_grp.add_argument('--predict',
                          metavar='INPUT_CSV',
                          default=None,
                          help='CSV with MSLG and SPA columns to score with a saved model')
    pred_grp.add_argument('--output',
                          metavar='OUTPUT_CSV',
                          default=None,
                          help='Output CSV path  (default: <input>_predictions.csv)')
    pred_grp.add_argument('--model',
                          default='model_v2.pkl',
                          help='Saved model bundle to use for prediction  (default: model_v2.pkl)')

    args = parser.parse_args()

    # ── Prediction mode ───────────────────────────────────────────────────────
    if args.predict is not None:
        stem     = os.path.splitext(args.predict)[0]
        out_path = args.output if args.output else stem + '_predictions.csv'
        predict_csv(args.model, args.predict, out_path)
        return

    # ── Training mode ─────────────────────────────────────────────────────────
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} examples  "
          f"(labels: {df['label'].value_counts().to_dict()})")

    print(f"\nBuilding feature matrix ({len(FEATURE_NAMES)} features)...")
    y     = df['label'].values
    X_all = build_feature_matrix(df)

    X_sel, sel_indices, sel_names = preselect_features(X_all, y)

    best_name, _ = evaluate_cv(X_sel, y, n_splits=args.cv)
    show_feature_importance(X_sel, y, best_name, feature_names=sel_names)

    if not args.no_save:
        save_model(X_sel, y, best_name, args.save_model, sel_indices=sel_indices)


if __name__ == '__main__':
    main()
