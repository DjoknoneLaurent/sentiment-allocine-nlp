# app/streamlit_app.py
# ============================================================
#  INTERFACE STREAMLIT — POINT D'ENTRÉE UNIQUE
#
#  Responsabilités :
#  · Configuration de la page et du thème
#  · Construction de l'UI (sidebar + zone principale)
#  · Appel à SentimentPredictor via st.cache_resource
#  · Affichage des résultats
#
#  Ce fichier ne contient AUCUNE logique ML.
#  Toute la logique est dans model_manager.py.
# ============================================================

import sys
from pathlib import Path

import streamlit as st

# ── Résolution des chemins ────────────────────────────────────
# Compatible exécution locale ET Streamlit Cloud
APP_DIR      = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

from model_manager import SentimentPredictor
from utils import clean_text, validate_input, format_prediction_result

CONFIG_PATH = PROJECT_ROOT / "deployment_config.json"

# ─────────────────────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# Doit être le premier appel Streamlit du script.
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Analyse de sentiment — Allociné",
    page_icon   = "🎬",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────
# CACHE DU PREDICTOR
# st.cache_resource : chargé une seule fois, partagé entre sessions.
# Sans ce cache, chaque interaction rechargerait les modèles.
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement des modèles...")
def get_predictor() -> SentimentPredictor:
    """
    Initialise le SentimentPredictor une seule fois.
    Le baseline est chargé immédiatement.
    CamemBERT est chargé à la première demande premium.
    """
    predictor = SentimentPredictor(CONFIG_PATH, PROJECT_ROOT)
    predictor._ensure_baseline_loaded()  # préchargement du baseline
    return predictor


# ─────────────────────────────────────────────────────────────
# CSS PERSONNALISÉ — style minimal et professionnel
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Carte de résultat */
.result-card {
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.1);
    margin: 1rem 0;
}
.result-positive { background: #f0fdf4; border-color: #86efac; }
.result-negative { background: #fef2f2; border-color: #fca5a5; }

/* Badge modèle */
.model-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 8px;
}
.badge-fast    { background: #dbeafe; color: #1d4ed8; }
.badge-premium { background: #fef3c7; color: #92400e; }
.badge-fallback{ background: #fee2e2; color: #991b1b; }

/* Métriques compactes */
.metric-row {
    display: flex;
    gap: 1.5rem;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}
.metric-item {
    display: flex;
    flex-direction: column;
    font-size: 0.85rem;
}
.metric-label { color: #6b7280; font-size: 0.75rem; }
.metric-value { font-weight: 600; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Sentiment Allociné")
    st.markdown("Analyse de sentiment sur critiques de films françaises.")
    st.divider()

    # Sélection du modèle
    st.markdown("### Modèle")
    model_choice_label = st.radio(
        label     = "Choisir le modèle :",
        options   = ["⚡ Rapide (LinearSVC)", "🧠 Premium (CamemBERT)"],
        index     = 0,
        help      = (
            "**Rapide** : LinearSVC TF-IDF. Instantané. F1=0.9424.\n\n"
            "**Premium** : CamemBERT fine-tuné. ~500ms. F1=0.9407. "
            "Meilleure généralisation sur formulations complexes."
        ),
    )
    model_choice = "fast" if "Rapide" in model_choice_label else "premium"

    st.divider()

    # Informations sur les modèles
    st.markdown("### À propos")
    with st.expander("Performances"):
        st.markdown("""
| Modèle | F1 | AUC |
|--------|-----|-----|
| LinearSVC | 0.9424 | 0.9874 |
| CamemBERT | 0.9407 | 0.9857 |

*Test set : 20 000 critiques Allociné*
        """)

    with st.expander("Critère métier"):
        st.markdown("""
Le seuil de décision est optimisé selon :

**coût = 3 × FN + 1 × FP**

Manquer un avis positif est 3× plus coûteux que signaler un faux positif.
Le seuil n'est jamais 0.5 par défaut.
        """)

    with st.expander("Exemple de critiques"):
        st.markdown("""
**Positive :**
> "Un chef-d'œuvre absolu. La mise en scène est magistrale, les acteurs transcendent leur rôle."

**Négative :**
> "Un film décevant, prévisible et sans âme. Je me suis ennuyé du début à la fin."
        """)

    st.divider()
    st.markdown(
        "Projet NLP · [GitHub](https://github.com) · "
        "Dataset [Allociné](https://huggingface.co/datasets/allocine)",
        unsafe_allow_html=False,
    )


# ─────────────────────────────────────────────────────────────
# ZONE PRINCIPALE
# ─────────────────────────────────────────────────────────────
st.markdown("# Analyse de sentiment de critiques cinéma")
st.markdown(
    "Colle une critique de film en français. "
    "Le modèle retourne le sentiment et son niveau de confiance."
)

# ── Zone de saisie ────────────────────────────────────────────
user_input = st.text_area(
    label       = "Critique de film :",
    placeholder = "Ex: Ce film est absolument magnifique, la réalisation est époustouflante...",
    height      = 160,
    max_chars   = 5000,
    key         = "user_input",
    label_visibility = "collapsed",
)

col_btn, col_clear, col_info = st.columns([2, 1, 4])

with col_btn:
    analyze_btn = st.button(
        "Analyser",
        type    = "primary",
        use_container_width = True,
    )
with col_clear:
    clear_btn = st.button(
        "Effacer",
        type    = "secondary",
        use_container_width = True,
    )

if clear_btn:
    st.rerun()

# ── Traitement et affichage du résultat ───────────────────────
if analyze_btn and user_input:

    # Validation de l'input
    is_valid, error_msg = validate_input(user_input)
    if not is_valid:
        st.error(f"⚠️ {error_msg}")
        st.stop()

    # Nettoyage
    cleaned_text = clean_text(user_input)

    # Chargement du predictor (mis en cache)
    predictor = get_predictor()

    # Prédiction avec spinner
    with st.spinner(
        "Analyse en cours..." if model_choice == "fast"
        else "CamemBERT en cours d'analyse (peut prendre ~30s au premier appel)..."
    ):
        try:
            result = predictor.predict(cleaned_text, model_choice=model_choice)
        except RuntimeError as e:
            st.error(f"❌ Erreur critique : {e}")
            st.info("Vérifie que les artefacts sont bien présents dans models/")
            st.stop()

    # ── Avertissement fallback ────────────────────────────────
    if result.is_fallback:
        st.warning(f"⚠️ {result.warning}")

    # ── Carte de résultat ─────────────────────────────────────
    is_positive    = result.label == 1
    card_class     = "result-positive" if is_positive else "result-negative"
    sentiment_icon = "😊" if is_positive else "😞"
    sentiment_text = "POSITIF" if is_positive else "NÉGATIF"
    confidence_pct = int(
        result.proba_pos * 100 if is_positive else result.proba_neg * 100
    )

    # Badge modèle
    if result.is_fallback:
        badge_class = "badge-fallback"
    elif model_choice == "premium":
        badge_class = "badge-premium"
    else:
        badge_class = "badge-fast"

    st.markdown(f"""
<div class="result-card {card_class}">
    <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem;">
        {sentiment_icon} {sentiment_text}
        <span class="model-badge {badge_class}">{result.model_used}</span>
    </div>
    <div class="metric-row">
        <div class="metric-item">
            <span class="metric-label">Confiance</span>
            <span class="metric-value">{confidence_pct}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">P(positif)</span>
            <span class="metric-value">{result.proba_pos:.4f}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">P(négatif)</span>
            <span class="metric-value">{result.proba_neg:.4f}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Seuil appliqué</span>
            <span class="metric-value">{result.threshold:.4f}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Latence</span>
            <span class="metric-value">{result.latency_ms} ms</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Jauges de probabilité ─────────────────────────────────
    st.markdown("#### Distribution des probabilités")
    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.metric(
            label = "Positif",
            value = f"{result.proba_pos:.1%}",
            delta = f"seuil : {result.threshold:.4f}",
        )
        st.progress(result.proba_pos)

    with col_neg:
        st.metric(
            label = "Négatif",
            value = f"{result.proba_neg:.1%}",
        )
        st.progress(result.proba_neg)

    # ── Niveau de confiance ───────────────────────────────────
    st.markdown("#### Niveau de confiance")
    conf_val = result.proba_pos if is_positive else result.proba_neg

    if conf_val >= 0.90:
        st.success(f"Très haute confiance ({conf_val:.1%}) — prédiction fiable.")
    elif conf_val >= 0.75:
        st.info(f"Bonne confiance ({conf_val:.1%}) — prédiction fiable.")
    elif conf_val >= 0.60:
        st.warning(
            f"Confiance modérée ({conf_val:.1%}) — le texte est ambigu. "
            "Essaie le modèle premium pour une analyse plus nuancée."
        )
    else:
        st.error(
            f"Faible confiance ({conf_val:.1%}) — critique très ambiguë. "
            "Les résultats sont moins fiables."
        )

    # ── Détail technique (expander) ───────────────────────────
    with st.expander("Détails techniques"):
        st.markdown(f"""
| Paramètre | Valeur |
|-----------|--------|
| Modèle utilisé | `{result.model_used}` |
| Seuil métier | `{result.threshold}` (optimisé : 3×FN + 1×FP) |
| Longueur texte original | {len(user_input)} caractères |
| Longueur texte nettoyé | {len(cleaned_text)} caractères |
| Temps d'inférence | {result.latency_ms} ms |
| Fallback activé | {"Oui" if result.is_fallback else "Non"} |
        """)

        st.markdown("**Texte transmis au modèle :**")
        st.code(cleaned_text[:500] + ("..." if len(cleaned_text) > 500 else ""))

elif analyze_btn and not user_input:
    st.warning("⚠️ Entre une critique de film avant d'analyser.")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.8rem;'>"
    "Projet NLP — Analyse de sentiment · Allociné · CamemBERT + LinearSVC"
    "</div>",
    unsafe_allow_html=True,
)
