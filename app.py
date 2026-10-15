from pathlib import Path
import html as html_lib

import altair as alt
import pandas as pd
import streamlit as st

from src.analytics import (
    OPTIONAL_COLUMN_CANDIDATES,
    analyze_reviews_dataframe,
    aspect_distribution_scores,
    build_time_trend,
    compute_kpis,
    emotion_distribution,
    filter_reviews,
    load_csv_input,
    priority_negative_reviews,
    sentiment_distribution,
)
from src.predict import predict_with_details

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"

EXAMPLE_REVIEWS = [
    "The product is bad!",
    "Terrible product... broke in 2 days!!!",
    "Excellent product, worth every rupee.",
    "Delivery was fast, but product quality was terrible.",
]

# ---------------------------------------------------------------------------
# Central theme constants
# ---------------------------------------------------------------------------
# Every color used by the dashboard is defined once here (Python) and mirrored
# into CSS custom properties so the palette stays cohesive and easy to tune.
PALETTE = {
    # surfaces
    "bg": "#0b1220",
    "surface": "#0f172a",
    "surface_2": "#111827",
    "surface_3": "#1e293b",
    "border": "#334155",
    "border_soft": "#1f2a3c",
    # text
    "text": "#f8fafc",
    "text_soft": "#cbd5e1",
    "text_muted": "#94a3b8",
    # sentiment - positive
    "positive": "#10b981",
    "positive_soft": "#34d399",
    "positive_bg": "rgba(16, 185, 129, 0.14)",
    "positive_text": "#6ee7b7",
    # sentiment - neutral
    "neutral": "#f59e0b",
    "neutral_soft": "#fbbf24",
    "neutral_bg": "rgba(245, 158, 11, 0.14)",
    "neutral_text": "#fcd34d",
    # sentiment - negative
    "negative": "#f43f5e",
    "negative_soft": "#fb7185",
    "negative_bg": "rgba(244, 63, 94, 0.14)",
    "negative_text": "#fda4af",
    # priority / urgent accent (used sparingly)
    "priority": "#ef4444",
    "priority_bg": "rgba(239, 68, 68, 0.12)",
    # generic accents
    "info": "#38bdf8",
    "accent": "#6366f1",
    "accent_soft": "#818cf8",
}

# Home page keeps its original palette mapping.
SENTIMENT_THEME = {
    "positive": {"primary": "#0f766e", "accent": "#14b8a6"},
    "negative": {"primary": "#be123c", "accent": "#fb7185"},
    "neutral": {"primary": "#1d4ed8", "accent": "#f59e0b"},
}

# Consistent semantic colors shared by charts, tables, and badges.
SENTIMENT_COLORS = {
    "positive": {
        "base": PALETTE["positive"],
        "bg": PALETTE["positive_bg"],
        "text": PALETTE["positive_text"],
    },
    "neutral": {
        "base": PALETTE["neutral"],
        "bg": PALETTE["neutral_bg"],
        "text": PALETTE["neutral_text"],
    },
    "negative": {
        "base": PALETTE["negative"],
        "bg": PALETTE["negative_bg"],
        "text": PALETTE["negative_text"],
    },
}

CHART_POSITIVE = PALETTE["positive"]
CHART_NEUTRAL = PALETTE["neutral"]
CHART_NEGATIVE = PALETTE["negative"]
CHART_LINE = PALETTE["info"]

REQUIRED_COLUMNS = [("review", "required")]
OPTIONAL_COLUMNS = [
    ("date", "optional"),
    ("product", "optional"),
    ("category", "optional"),
    ("rating", "optional"),
    ("customer", "optional"),
]

st.set_page_config(
    page_title="Customer Sentiment Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
_CSS_VARS = f"""
:root {{
  --bg: {PALETTE['bg']};
  --surface: {PALETTE['surface']};
  --surface-2: {PALETTE['surface_2']};
  --surface-3: {PALETTE['surface_3']};
  --border: {PALETTE['border']};
  --border-soft: {PALETTE['border_soft']};
  --text: {PALETTE['text']};
  --text-soft: {PALETTE['text_soft']};
  --text-muted: {PALETTE['text_muted']};
  --positive: {PALETTE['positive']};
  --positive-soft: {PALETTE['positive_soft']};
  --positive-bg: {PALETTE['positive_bg']};
  --neutral: {PALETTE['neutral']};
  --neutral-soft: {PALETTE['neutral_soft']};
  --neutral-bg: {PALETTE['neutral_bg']};
  --negative: {PALETTE['negative']};
  --negative-soft: {PALETTE['negative_soft']};
  --negative-bg: {PALETTE['negative_bg']};
  --priority: {PALETTE['priority']};
  --priority-bg: {PALETTE['priority_bg']};
  --info: {PALETTE['info']};
  --accent: {PALETTE['accent']};
  --accent-soft: {PALETTE['accent_soft']};
}}
"""

_CSS_MAIN = """
.stApp {
  background:
    radial-gradient(1100px 560px at 12% -12%, rgba(37, 99, 235, 0.10), transparent 62%),
    radial-gradient(900px 480px at 92% -4%, rgba(16, 185, 129, 0.06), transparent 58%),
    var(--bg);
}
.block-container {
  padding-top: 2.1rem;
  padding-bottom: 3rem;
  max-width: 1440px;
}

/* Hero + home page cards (preserved from the earlier redesign) */
.subtitle {
  color: var(--text-muted);
  margin-top: -8px;
}
.hero-card {
  border-radius: 16px;
  padding: 18px 20px;
  border: 1px solid var(--border);
  background: linear-gradient(135deg, #0f172a 0%, #172554 100%);
  color: var(--text);
  margin-bottom: 14px;
}
.card {
  border-radius: 16px;
  padding: 16px;
  border: 1px solid var(--border);
  background: var(--surface);
}
.sentiment-card {
  border-radius: 16px;
  padding: 18px 20px;
  color: var(--text);
  margin-top: 10px;
  border: 1px solid var(--border);
  box-shadow: 0 12px 28px rgba(2, 8, 23, 0.35);
}
.result-empty {
  background: linear-gradient(160deg, #111827 0%, #1e293b 100%);
  border: 1px dashed #475569;
  border-radius: 12px;
  padding: 16px;
  color: #e2e8f0;
}
.result-empty h4 {
  margin: 0 0 4px 0;
}
.result-empty p {
  margin: 0;
  color: #cbd5e1;
}
.cue-chip {
  display: inline-block;
  margin: 4px 6px 0 0;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid #64748b;
  background: rgba(30, 41, 59, 0.9);
  color: var(--text);
  font-size: 12px;
}
.probability-row {
  margin-top: 10px;
}
.probability-label {
  display: flex;
  justify-content: space-between;
  color: var(--text-soft);
  font-size: 13px;
  margin-bottom: 4px;
}
.probability-track {
  width: 100%;
  height: 10px;
  border-radius: 999px;
  background: #1e293b;
  overflow: hidden;
}
.probability-fill {
  height: 100%;
  border-radius: 999px;
}

/* Section headings */
.section-head {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin: 22px 0 12px 0;
}
.section-bar {
  width: 4px;
  align-self: stretch;
  min-height: 34px;
  border-radius: 999px;
  background: linear-gradient(180deg, var(--info), var(--accent));
  flex: 0 0 4px;
}
.section-head-title {
  color: var(--text);
  font-size: 19px;
  font-weight: 700;
  line-height: 1.2;
  margin: 0;
}
.section-sub {
  display: block;
  color: var(--text-muted);
  font-size: 13px;
  margin-top: 2px;
}

/* Generic surface panel */
.surface-panel {
  border: 1px solid var(--border);
  border-radius: 16px;
  background: linear-gradient(160deg, rgba(17, 24, 39, 0.88), rgba(15, 23, 42, 0.96));
  padding: 16px 18px;
  box-shadow: 0 12px 26px rgba(2, 8, 23, 0.30);
}
.surface-panel .panel-title {
  color: var(--text);
  font-weight: 700;
  font-size: 15px;
  margin-bottom: 6px;
}
.surface-panel .panel-body {
  color: var(--text-soft);
  font-size: 13.5px;
  line-height: 1.55;
}
.surface-panel .panel-body p {
  margin: 0 0 8px 0;
}
.info-panel { border-left: 3px solid var(--info); }
.success-panel { border-left: 3px solid var(--positive); }
.error-panel { border-left: 3px solid var(--negative); }
.priority-panel { border-left: 3px solid var(--priority); }
.empty-panel {
  border: 1px dashed #475569;
  border-radius: 16px;
  background: linear-gradient(160deg, rgba(17, 24, 39, 0.75), rgba(15, 23, 42, 0.9));
  padding: 26px 22px;
  text-align: center;
  color: var(--text-soft);
}
.empty-panel .panel-title { color: var(--text); font-size: 16px; }
.empty-panel .panel-body { color: var(--text-muted); }

/* Column / schema chips */
.col-chip {
  display: inline-block;
  margin: 3px 6px 0 0;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  border: 1px solid var(--border);
  background: rgba(30, 41, 59, 0.75);
  color: var(--text-soft);
}
.col-chip.required {
  border-color: rgba(244, 63, 94, 0.55);
  background: var(--negative-bg);
  color: var(--negative-text);
  font-weight: 600;
}
.col-chip.optional {
  border-color: rgba(56, 189, 248, 0.40);
  background: rgba(56, 189, 248, 0.10);
  color: #bae6fd;
}
.col-chip.detected {
  border-color: rgba(16, 185, 129, 0.55);
  background: var(--positive-bg);
  color: var(--positive-text);
  font-weight: 600;
}
.col-chip.missing {
  opacity: 0.65;
}

/* KPI cards */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 14px;
  margin: 6px 0 8px 0;
}
.kpi-card {
  position: relative;
  background: linear-gradient(160deg, #111827, #0f172a);
  border: 1px solid var(--border);
  border-top: 3px solid var(--kpi-accent, var(--info));
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 10px 24px rgba(2, 8, 23, 0.35);
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}
.kpi-card:hover,
.kpi-card:focus-within {
  transform: translateY(-2px);
  border-color: var(--kpi-accent, var(--info));
  box-shadow: 0 16px 30px rgba(2, 8, 23, 0.5);
}
.kpi-label {
  color: var(--text-muted);
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}
.kpi-value {
  color: var(--text);
  font-size: 27px;
  font-weight: 700;
  line-height: 1.1;
  margin: 7px 0 3px 0;
}
.kpi-hint {
  color: var(--text-soft);
  font-size: 12px;
}

/* Chart containers */
.chart-card {
  border: 1px solid var(--border);
  border-radius: 16px;
  background: linear-gradient(160deg, rgba(17, 24, 39, 0.72), rgba(15, 23, 42, 0.9));
  padding: 12px 14px 4px 14px;
  box-shadow: 0 12px 26px rgba(2, 8, 23, 0.28);
  margin-bottom: 8px;
}

/* Priority review items */
.priority-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.priority-item {
  position: relative;
  border: 1px solid var(--border);
  border-left: 4px solid var(--item-accent, var(--priority));
  border-radius: 14px;
  background: linear-gradient(150deg, rgba(30, 41, 59, 0.55), rgba(15, 23, 42, 0.95));
  padding: 14px 16px;
  box-shadow: 0 10px 24px rgba(2, 8, 23, 0.32);
  transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
}
.priority-item:hover {
  transform: translateY(-1px);
  box-shadow: 0 14px 30px rgba(2, 8, 23, 0.5);
}
.priority-review {
  color: var(--text);
  font-size: 14.5px;
  line-height: 1.5;
  margin: 0 0 10px 0;
}
.priority-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 10px;
  align-items: center;
}
.meta-pill {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 11.5px;
  font-weight: 600;
  border: 1px solid var(--border);
  background: rgba(30, 41, 59, 0.8);
  color: var(--text-soft);
}
.meta-pill.priority-high {
  border-color: rgba(239, 68, 68, 0.6);
  background: var(--priority-bg);
  color: #fecaca;
}
.meta-pill.priority-med {
  border-color: rgba(245, 158, 11, 0.5);
  background: var(--neutral-bg);
  color: var(--neutral-text);
}
.meta-pill.urgency {
  border-color: rgba(244, 63, 94, 0.55);
  background: var(--negative-bg);
  color: var(--negative-text);
}

/* Tabs, buttons, progress (home page look preserved) */
.stTabs [data-baseweb="tab-list"] {
  gap: 10px;
}
.stTabs [data-baseweb="tab"] {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 12px;
  color: var(--text-soft);
  padding: 8px 14px;
}
.stTabs [aria-selected="true"] {
  background: #1d4ed8 !important;
  border-color: #60a5fa !important;
  color: #eff6ff !important;
}
.stButton > button {
  border-radius: 10px !important;
}
.stButton > button:not([kind="primary"]) {
  background: var(--surface-3) !important;
  border: 1px solid #475569 !important;
  color: #e2e8f0 !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #2563eb, #0ea5e9) !important;
  border: 1px solid #93c5fd !important;
  color: #eff6ff !important;
}
div[data-testid="stMetric"] {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px;
  background: var(--surface);
}
div[data-testid="stProgressBar"] > div > div > div {
  background: linear-gradient(90deg, #06b6d4, #3b82f6) !important;
}

/* Widget labels */
[data-testid="stWidgetLabel"] p {
  color: var(--text-soft) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
}

/* Text inputs / areas */
div[data-baseweb="input"],
div[data-baseweb="textarea"],
div[data-baseweb="select"] > div {
  background-color: rgba(15, 23, 42, 0.9) !important;
  border-color: var(--border) !important;
  border-radius: 10px !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] input,
div[data-baseweb="select"] span {
  color: var(--text) !important;
}
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder {
  color: var(--text-muted) !important;
}
div[data-baseweb="input"]:focus-within,
div[data-baseweb="textarea"]:focus-within,
div[data-baseweb="select"] > div:focus-within {
  border-color: var(--info) !important;
  box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.16) !important;
}

/* Multiselect tags */
div[data-baseweb="tag"] {
  background-color: rgba(99, 102, 241, 0.22) !important;
  border: 1px solid rgba(129, 140, 248, 0.5) !important;
  border-radius: 8px !important;
}
div[data-baseweb="tag"] span {
  color: #e0e7ff !important;
}

/* Dropdown menus */
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"],
ul[data-baseweb="menu"] {
  background-color: var(--surface-2) !important;
}
li[role="option"] {
  color: var(--text-soft) !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
  background-color: rgba(56, 189, 248, 0.16) !important;
  color: var(--text) !important;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
  background: linear-gradient(160deg, rgba(30, 41, 59, 0.72), rgba(15, 23, 42, 0.92));
  border: 1.5px dashed var(--border);
  border-radius: 16px;
  padding: 22px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--info);
  box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.14);
}
[data-testid="stFileUploaderDropzoneInstructions"] {
  color: var(--text-soft);
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
  color: var(--text) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small {
  color: var(--text-muted) !important;
}
[data-testid="stFileUploaderDropzone"] button {
  background: rgba(56, 189, 248, 0.14) !important;
  border: 1px solid rgba(56, 189, 248, 0.55) !important;
  color: #e0f2fe !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
  background: rgba(56, 189, 248, 0.24) !important;
  border-color: var(--info) !important;
}

/* Bordered containers used as dashboard panels */
.st-key-filters_panel [data-testid="stVerticalBlock"],
.st-key-csv_upload_panel [data-testid="stVerticalBlock"] {
  border-color: var(--border) !important;
  border-radius: 16px !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 12px 26px rgba(2, 8, 23, 0.28);
}

/* Alerts */
[data-testid="stAlert"] {
  border-radius: 12px;
}

/* Divider */
hr {
  border-color: var(--border-soft);
}

/* Responsive KPI grid */
@media (max-width: 1180px) {
  .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 760px) {
  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 470px) {
  .kpi-grid { grid-template-columns: 1fr; }
}
"""

st.markdown(
    "<style>" + _CSS_VARS + _CSS_MAIN + "</style>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Reusable UI helpers
# ---------------------------------------------------------------------------
def section_title(title: str, subtitle: str = "") -> None:
    sub = f"<span class='section-sub'>{subtitle}</span>" if subtitle else ""
    st.markdown(
        f"""
        <div class='section-head'>
          <span class='section-bar'></span>
          <div>
            <div class='section-head-title'>{title}</div>
            {sub}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(kind: str, title: str, body: str) -> str:
    """Return HTML for a styled state panel. kind: info/success/error/empty/priority."""
    return (
        f"<div class='surface-panel {kind}-panel'>"
        f"<div class='panel-title'>{title}</div>"
        f"<div class='panel-body'>{body}</div>"
        f"</div>"
    )


def chip(text: str, kind: str = "optional") -> str:
    return f"<span class='col-chip {kind}'>{html_lib.escape(text)}</span>"


def kpi_card(label: str, value: str, accent: str, hint: str = "") -> str:
    return (
        f"<div class='kpi-card' style='--kpi-accent:{accent};'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-hint'>{hint}</div>"
        f"</div>"
    )


def render_kpi_grid(cards: list[str]) -> None:
    st.markdown("<div class='kpi-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def _apply_chart_theme(chart: alt.Chart) -> alt.Chart:
    """Apply the shared dark, readable chart theme."""
    return (
        chart.configure_view(strokeOpacity=0, fill=None)
        .configure_axis(
            gridColor="#1e293b",
            domainColor="#334155",
            tickColor="#334155",
            labelColor=PALETTE["text_soft"],
            titleColor=PALETTE["text_muted"],
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=PALETTE["text_soft"],
            titleColor=PALETTE["text_muted"],
            orient="bottom",
            direction="horizontal",
        )
        .configure_text(color=PALETTE["text_soft"])
    )


def render_chart(chart: alt.Chart) -> None:
    st.altair_chart(_apply_chart_theme(chart), use_container_width=True)


def _style_sentiment_cell(value: object) -> str:
    colors = SENTIMENT_COLORS.get(str(value).strip().lower())
    if not colors:
        return ""
    return f"background-color:{colors['bg']}; color:{colors['text']}; font-weight:600;"


def _style_priority_cell(value: object) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return ""
    if score >= 0.9:
        return f"color:{PALETTE['priority']}; font-weight:700;"
    if score >= 0.7:
        return f"color:{PALETTE['neutral_soft']}; font-weight:600;"
    return f"color:{PALETTE['text_muted']};"


def _style_cells(styler, func, subset):
    """Element-wise Styler styling that works on pandas 2.0 (applymap) and 2.1+ (map)."""
    mapper = getattr(styler, "map", None) or getattr(styler, "applymap")
    return mapper(func, subset=subset)


def build_explorer_table(df: pd.DataFrame):
    """Build a readable, color-coded view of the review explorer (display only)."""
    display = df.copy()

    if "sentiment_score_0_10" in display.columns:
        display["sentiment_score_0_10"] = pd.to_numeric(
            display["sentiment_score_0_10"], errors="coerce"
        )
    if "confidence" in display.columns:
        display["confidence"] = pd.to_numeric(display["confidence"], errors="coerce") * 100.0
    if "priority_score" in display.columns:
        display["priority_score"] = pd.to_numeric(display["priority_score"], errors="coerce")
    if "date" in display.columns:
        display["date"] = display["date"].astype(str)

    column_order = [
        "review",
        "sentiment",
        "sentiment_score_0_10",
        "confidence",
        "dominant_emotion",
        "priority_score",
    ]
    for optional in ["product", "category", "rating", "customer", "date"]:
        if optional in display.columns:
            column_order.append(optional)
    display = display[column_order].rename(
        columns={
            "review": "Review",
            "sentiment": "Sentiment",
            "sentiment_score_0_10": "Sentiment score",
            "confidence": "Confidence",
            "dominant_emotion": "Emotion",
            "priority_score": "Priority",
            "product": "Product",
            "category": "Category",
            "rating": "Rating",
            "customer": "Customer",
            "date": "Date",
        }
    )

    format_map = {
        "Sentiment score": "{:.1f}",
        "Confidence": "{:.1f}%",
        "Priority": "{:.2f}",
    }
    styler = display.style.format(format_map, na_rep="—")
    styler = _style_cells(styler, _style_sentiment_cell, ["Sentiment"])
    if "Priority" in display.columns:
        styler = _style_cells(styler, _style_priority_cell, ["Priority"])
    return styler


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class='hero-card'>
      <h2 style='margin:0;'>📊 Customer Sentiment Intelligence Dashboard</h2>
      <p class='subtitle'>Local-first sentiment analytics with transparent heuristic/demo layers for aspects and emotions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.warning("No trained model found yet.")
    st.markdown("Train first with `python src/train.py` from the repository root.")
    st.stop()


@st.cache_data(show_spinner=False)
def _cached_batch_analysis(file_bytes: bytes) -> pd.DataFrame:
    source_df = load_csv_input(file_bytes)
    return analyze_reviews_dataframe(source_df)


tab_single, tab_dashboard = st.tabs(["Single Review Analysis", "CSV Intelligence Dashboard"])

with tab_single:
    if "review_input" not in st.session_state:
        st.session_state.review_input = ""

    input_col, output_col = st.columns([1.1, 1.0], gap="large")

    with input_col:
        st.subheader("Analyze one review")
        st.session_state.review_input = st.text_area(
            "Review text",
            value=st.session_state.review_input,
            placeholder="Example: Delivery was fast, but product quality was terrible.",
            height=180,
        )

        st.markdown("**Try examples**")
        button_cols = st.columns(2)
        for i, example in enumerate(EXAMPLE_REVIEWS):
            if button_cols[i % 2].button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.review_input = example
                st.rerun()

        analyze_clicked = st.button("Analyze sentiment", type="primary", use_container_width=True)

    with output_col:
        st.subheader("Result")
        if analyze_clicked:
            review = st.session_state.review_input.strip()
            if not review:
                st.error("Please enter a review before analyzing.")
            else:
                try:
                    with st.spinner("Analyzing review..."):
                        result = predict_with_details(review)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                else:
                    sentiment = str(result["sentiment"])
                    confidence = float(result["confidence"])
                    probabilities = result["probabilities"]
                    signals = result["signals"]

                    sentiment_style = SENTIMENT_THEME.get(sentiment, {"primary": "#1d4ed8", "accent": "#60a5fa"})
                    st.markdown(
                        f"""
                        <div class='sentiment-card' style='background:linear-gradient(140deg, {sentiment_style["primary"]}, {sentiment_style["accent"]});'>
                          <h3 style='margin:0;'>Sentiment: {sentiment.title()}</h3>
                          <p style='margin:8px 0 0 0; color:#f8fafc;'>Model confidence estimate: {confidence:.1%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.progress(min(max(confidence, 0.0), 1.0), text="Confidence")
                    prob_cols = st.columns(3)
                    for i, label in enumerate(["positive", "neutral", "negative"]):
                        value = float(probabilities.get(label, 0.0))
                        prob_cols[i].metric(label.title(), f"{value:.1%}")

                    prob_colors = {
                        "positive": "#14b8a6",
                        "neutral": "#60a5fa",
                        "negative": "#fb7185",
                    }
                    for label in ["positive", "neutral", "negative"]:
                        value = max(0.0, min(float(probabilities.get(label, 0.0)), 1.0))
                        st.markdown(
                            f"""
                            <div class='probability-row'>
                              <div class='probability-label'>
                                <span>{label.title()}</span>
                                <span>{value:.1%}</span>
                              </div>
                              <div class='probability-track'>
                                <div class='probability-fill' style='width:{value * 100:.1f}%; background:{prob_colors[label]};'></div>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    if signals:
                        st.markdown("**Signal/Cue words**")
                        for label in ["positive", "neutral", "negative"]:
                            words = signals.get(label)
                            if words:
                                st.markdown(
                                    f"{label.title()}: "
                                    + " ".join(f"<span class='cue-chip'>{word}</span>" for word in words),
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.caption("No clear cue words matched; prediction relies on broader text patterns.")

                    st.markdown(
                        "<div class='result-empty' style='margin-top:12px; border-style:solid;'>Aspect and emotion layers are available in the CSV dashboard tab as transparent heuristic/demo analytics.</div>",
                        unsafe_allow_html=True,
                    )
        if not analyze_clicked and st.session_state.review_input.strip():
            st.markdown(
                "<div class='result-empty'><h4>Ready to analyze</h4><p>Click <b>Analyze sentiment</b> to generate a sentiment card, confidence, probabilities, and cue words.</p></div>",
                unsafe_allow_html=True,
            )
        elif not st.session_state.review_input.strip() and not analyze_clicked:
            st.markdown(
                "<div class='result-empty'><h4>Awaiting review text</h4><p>Type a customer review or choose an example to see sentiment, confidence, probability breakdown, and cue words.</p></div>",
                unsafe_allow_html=True,
            )

with tab_dashboard:
    section_title(
        "Upload reviews CSV",
        "Drag and drop a CSV file, or browse to select one. Every review row is scored with the local model.",
    )

    upload_col, schema_col = st.columns([1.35, 1.0], gap="large")

    with upload_col:
        with st.container(border=True, key="csv_upload_panel"):
            st.markdown("**Review data file**")
            uploaded_file = st.file_uploader(
                "Drop your CSV here",
                type=["csv"],
                accept_multiple_files=False,
                help="A single .csv file with a review column.",
            )

    with schema_col:
        required_chips = "".join(chip(name, "required") for name, _ in REQUIRED_COLUMNS)
        optional_chips = "".join(chip(name, "optional") for name, _ in OPTIONAL_COLUMNS)
        st.markdown(
            panel(
                "info",
                "Expected columns",
                "<p>At minimum, include the <b>review</b> column (matched case-insensitively). "
                "These optional columns unlock richer analytics when present:</p>"
                f"<div style='margin:2px 0 10px 0;'>{required_chips}</div>"
                f"<div>{optional_chips}</div>"
                "<p style='margin-top:10px;' class='panel-body'>Optional columns are auto-detected "
                "(e.g. <code>timestamp</code>, <code>product_name</code>, <code>stars</code>). "
                "Missing optional columns are handled gracefully.</p>",
            ),
            unsafe_allow_html=True,
        )

    if uploaded_file is None:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown(
            panel(
                "empty",
                "No CSV uploaded yet",
                "<p>Add your reviews file to generate KPI cards, sentiment and emotion charts, "
                "a time trend, an aspect breakdown, and a priority queue for urgent negative reviews.</p>"
                "<p style='margin-top:10px;'>Example header row:</p>"
                f"<div>{chip('review', 'required')}{chip('date', 'optional')}{chip('product', 'optional')}"
                f"{chip('category', 'optional')}{chip('rating', 'optional')}{chip('customer', 'optional')}</div>",
            ),
            unsafe_allow_html=True,
        )
    else:
        try:
            bytes_data = uploaded_file.getvalue()
            with st.spinner("Analyzing reviews with the local model…"):
                analysis_df = _cached_batch_analysis(bytes_data)
        except Exception as exc:
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            st.markdown(
                panel(
                    "error",
                    "We couldn't read that CSV",
                    f"<p>{html_lib.escape(str(exc))}</p>"
                    "<p>Please make sure the file is a valid CSV with a <b>review</b> column "
                    "and at least one non-empty review.</p>",
                ),
                unsafe_allow_html=True,
            )
        else:
            detected = [name for name in OPTIONAL_COLUMN_CANDIDATES if name in analysis_df.columns]
            detected_chips = "".join(chip(name, "detected") for name in detected)
            missing_chips = "".join(
                chip(name, "missing") for name in OPTIONAL_COLUMN_CANDIDATES if name not in detected
            )
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            st.markdown(
                panel(
                    "success",
                    f"Loaded {len(analysis_df)} reviews",
                    f"<div style='margin-bottom:6px;'>Detected optional columns: "
                    f"{detected_chips if detected_chips else '<span class=\"panel-body\'>none</span>'}</div>"
                    f"<div>Not detected: {missing_chips if missing_chips else '<span class=\"panel-body\'>none</span>'}</div>",
                ),
                unsafe_allow_html=True,
            )

            kpis = compute_kpis(analysis_df)

            score = kpis["avg_sentiment_score"]
            score_accent = (
                PALETTE["positive"] if score >= 7 else PALETTE["neutral"] if score >= 4 else PALETTE["negative"]
            )
            render_kpi_grid(
                [
                    kpi_card("Total Reviews", f"{kpis['total_reviews']:,}", PALETTE["info"], "reviews analyzed"),
                    kpi_card("Avg Sentiment", f"{kpis['avg_sentiment_score']:.2f}", score_accent, "0 = negative · 10 = positive"),
                    kpi_card("Positive", f"{kpis['positive_pct']:.1f}%", PALETTE["positive"], "share of reviews"),
                    kpi_card("Negative", f"{kpis['negative_pct']:.1f}%", PALETTE["negative"], "share of reviews"),
                    kpi_card("Avg Confidence", f"{kpis['avg_confidence']:.1f}%", PALETTE["accent"], "model estimate"),
                ]
            )

            # --- Charts -----------------------------------------------------
            chart_col_1, chart_col_2 = st.columns(2, gap="large")
            with chart_col_1:
                section_title("Sentiment distribution", "Reviews by predicted sentiment")
                dist = sentiment_distribution(analysis_df)
                dist_df = pd.DataFrame(
                    {"sentiment": dist.index.astype(str), "reviews": dist.to_numpy()}
                )
                bars = (
                    alt.Chart(dist_df)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=56)
                    .encode(
                        x=alt.X(
                            "sentiment:N",
                            sort=["positive", "neutral", "negative"],
                            axis=alt.Axis(title=None, labelAngle=0),
                        ),
                        y=alt.Y("reviews:Q", axis=alt.Axis(title="Reviews", tickMinStep=1)),
                        color=alt.Color(
                            "sentiment:N",
                            scale=alt.Scale(
                                domain=["positive", "neutral", "negative"],
                                range=[CHART_POSITIVE, CHART_NEUTRAL, CHART_NEGATIVE],
                            ),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("sentiment:N", title="Sentiment"),
                            alt.Tooltip("reviews:Q", title="Reviews"),
                        ],
                    )
                )
                bar_labels = bars.mark_text(dy=-8, color=PALETTE["text"], fontSize=12, fontWeight="bold").encode(
                    text=alt.Text("reviews:Q", format="d")
                )
                render_chart((bars + bar_labels).properties(height=280))

            with chart_col_2:
                section_title("Emotion distribution", "Heuristic/demo emotion cues")
                emotions = emotion_distribution(analysis_df)
                if emotions.empty:
                    st.markdown(
                        panel("empty", "No emotion cues detected", "No emotion keywords were found in the current data."),
                        unsafe_allow_html=True,
                    )
                else:
                    emotion_df = pd.DataFrame(
                        {"emotion": emotions.index.astype(str), "count": emotions.to_numpy()}
                    )
                    emotion_chart = (
                        alt.Chart(emotion_df)
                        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, size=20)
                        .encode(
                            y=alt.Y("emotion:N", sort="-x", axis=alt.Axis(title=None)),
                            x=alt.X("count:Q", axis=alt.Axis(title="Mentions", tickMinStep=1)),
                            color=alt.Color(
                                "count:Q",
                                scale=alt.Scale(range=["#1e40af", PALETTE["info"]]),
                                legend=None,
                            ),
                            tooltip=[
                                alt.Tooltip("emotion:N", title="Emotion"),
                                alt.Tooltip("count:Q", title="Mentions"),
                            ],
                        )
                        .properties(height=max(240, 34 * len(emotion_df)))
                    )
                    render_chart(emotion_chart)

            section_title("Aspect sentiment", "Average sentiment score per aspect (0–10, heuristic/demo)")
            aspect_scores = aspect_distribution_scores(analysis_df)
            if aspect_scores.empty:
                st.markdown(
                    panel("empty", "No aspect mentions detected", "No configured aspect keywords were found in the current data."),
                    unsafe_allow_html=True,
                )
            else:
                aspect_df = pd.DataFrame(
                    {"aspect": aspect_scores.index.astype(str), "score": aspect_scores.to_numpy()}
                )
                aspect_mid_rule = (
                    alt.Chart(pd.DataFrame({"x": [5.0]}))
                    .mark_rule(color="#475569", strokeDash=[4, 4])
                    .encode(x=alt.X("x:Q"))
                )
                aspect_chart = (
                    alt.Chart(aspect_df)
                    .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, size=20)
                    .encode(
                        y=alt.Y("aspect:N", sort="-x", axis=alt.Axis(title=None)),
                        x=alt.X(
                            "score:Q",
                            scale=alt.Scale(domain=[0, 10]),
                            axis=alt.Axis(title="Sentiment score (0-10)", values=[0, 2, 4, 6, 8, 10]),
                        ),
                        color=alt.Color(
                            "score:Q",
                            scale=alt.Scale(
                                domain=[0, 5, 10],
                                range=[CHART_NEGATIVE, CHART_NEUTRAL, CHART_POSITIVE],
                            ),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("aspect:N", title="Aspect"),
                            alt.Tooltip("score:Q", title="Score", format=".2f"),
                        ],
                    )
                    .properties(height=max(240, 40 * len(aspect_df)))
                )
                render_chart(aspect_mid_rule + aspect_chart)

            section_title("Time trend", "Average sentiment score over time (requires a usable date column)")
            trend_df = build_time_trend(analysis_df)
            if trend_df.empty:
                st.markdown(
                    panel(
                        "empty",
                        "Trend unavailable",
                        "No usable date/time column was detected, so the trend view is hidden. "
                        f"Add a {chip('date', 'optional')} column to enable it.",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                trend_plot = trend_df.copy()
                trend_plot["date"] = pd.to_datetime(trend_plot["date"])
                trend_chart = (
                    alt.Chart(trend_plot)
                    .mark_line(
                        color=CHART_LINE,
                        strokeWidth=2.5,
                        interpolate="monotone",
                        point=alt.OverlayMarkDef(color=CHART_LINE, filled=True, size=55),
                    )
                    .encode(
                        x=alt.X("date:T", axis=alt.Axis(title=None, format="%b %d", labelAngle=-30)),
                        y=alt.Y(
                            "avg_sentiment_score:Q",
                            scale=alt.Scale(domain=[0, 10]),
                            axis=alt.Axis(title="Avg sentiment (0-10)"),
                        ),
                        tooltip=[
                            alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
                            alt.Tooltip("avg_sentiment_score:Q", title="Avg score", format=".2f"),
                            alt.Tooltip("review_count:Q", title="Reviews"),
                        ],
                    )
                    .properties(height=300)
                )
                render_chart(trend_chart)

            # --- Review explorer -------------------------------------------
            section_title("Review explorer", "Search and filter individual reviews")

            with st.container(border=True, key="filters_panel"):
                st.markdown("<div class='panel-title' style='color:#e2e8f0;font-weight:700;font-size:14px;'>Filters</div>", unsafe_allow_html=True)
                filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns(4, gap="medium")
                selected_sentiments = filter_col_1.multiselect(
                    "Sentiment",
                    options=["positive", "neutral", "negative"],
                    placeholder="All sentiments",
                )
                aspect_options = sorted(
                    {aspect for tags in analysis_df["aspect_tags"] for aspect in tags}
                )
                selected_aspects = filter_col_2.multiselect(
                    "Aspect",
                    options=aspect_options,
                    placeholder="All aspects",
                )
                emotion_options = sorted(
                    {emotion for tags in analysis_df["emotion_tags"] for emotion in tags}
                )
                selected_emotions = filter_col_3.multiselect(
                    "Emotion",
                    options=emotion_options,
                    placeholder="All emotions",
                )
                search_text = filter_col_4.text_input("Search reviews", placeholder="Search text…")

                product_selection: list[str] = []
                category_selection: list[str] = []
                date_selection = None
                has_product = "product" in analysis_df.columns
                has_category = "category" in analysis_df.columns
                has_dates = "parsed_date" in analysis_df.columns and analysis_df["parsed_date"].notna().any()
                if has_product or has_category or has_dates:
                    filter_col_5, filter_col_6, filter_col_7 = st.columns(3, gap="medium")
                    if has_product:
                        product_values = sorted(analysis_df["product"].dropna().astype(str).unique().tolist())
                        product_selection = filter_col_5.multiselect(
                            "Product",
                            options=product_values,
                            placeholder="All products",
                        )
                    if has_category:
                        category_values = sorted(analysis_df["category"].dropna().astype(str).unique().tolist())
                        category_selection = filter_col_6.multiselect(
                            "Category",
                            options=category_values,
                            placeholder="All categories",
                        )
                    if has_dates:
                        valid_dates = analysis_df["parsed_date"].dropna()
                        date_selection = filter_col_7.date_input(
                            "Date range",
                            value=None,
                            min_value=valid_dates.min().date(),
                            max_value=valid_dates.max().date(),
                            help="Leave empty to include reviews regardless of date.",
                        )

            filtered_df = filter_reviews(
                analysis_df,
                search_text=search_text,
                sentiments=selected_sentiments,
                aspects=selected_aspects,
                emotions=selected_emotions,
                products=product_selection,
                categories=category_selection,
            )

            # Optional, additive date filter (defaults to "all dates").
            if date_selection and len(date_selection) == 2 and "parsed_date" in filtered_df.columns:
                start_date, end_date = date_selection
                date_mask = filtered_df["parsed_date"].dt.date.between(start_date, end_date)
                filtered_df = filtered_df[date_mask].reset_index(drop=True)

            if filtered_df.empty:
                st.markdown(
                    panel(
                        "empty",
                        "No reviews match your filters",
                        "Try clearing one or more filters, widening the date range, or using a different search term. "
                        "All filters are combinable (AND).",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.caption(f"Showing {len(filtered_df)} of {len(analysis_df)} reviews")
                st.dataframe(
                    build_explorer_table(filtered_df),
                    use_container_width=True,
                    hide_index=True,
                )

                # --- Priority negative reviews -----------------------------
                section_title(
                    "Highest-priority negative reviews",
                    "Transparent heuristic = negative sentiment + model confidence + severity cue hits (demo only).",
                )
                priority_df = priority_negative_reviews(filtered_df, limit=10)
                if priority_df.empty:
                    st.markdown(
                        panel(
                            "success",
                            "No negative reviews in view",
                            "No negative reviews match the current filters — nothing needs urgent attention.",
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    items = []
                    for _, row in priority_df.iterrows():
                        priority_score = float(row.get("priority_score", 0.0) or 0.0)
                        confidence = float(row.get("confidence", 0.0) or 0.0)
                        severity = int(row.get("severity_hits", 0) or 0)
                        emotion = str(row.get("dominant_emotion", "none") or "none")
                        accent = PALETTE["priority"] if severity >= 2 else PALETTE["negative_soft"]
                        priority_class = "priority-high" if priority_score >= 0.9 else "priority-med"
                        severity_chip = (
                            f"<span class='meta-pill urgency'>⛔ {severity} severity cue"
                            f"{'s' if severity != 1 else ''}</span>"
                            if severity > 0
                            else "<span class='meta-pill'>no severity cues</span>"
                        )
                        items.append(
                            "<div class='priority-item' style='--item-accent:" + accent + ";'>"
                            "<p class='priority-review'>"
                            + html_lib.escape(str(row.get("review", "")))
                            + "</p>"
                            "<div class='priority-meta'>"
                            + f"<span class='meta-pill {priority_class}'>⚠ Priority {priority_score:.2f}</span>"
                            + f"<span class='meta-pill'>Confidence {confidence:.1%}</span>"
                            + f"<span class='meta-pill'>Emotion: {html_lib.escape(emotion)}</span>"
                            + severity_chip
                            + "</div></div>"
                        )
                    count = len(priority_df)
                    st.markdown(
                        panel(
                            "priority",
                            f"{count} review{'s' if count != 1 else ''} "
                            f"{'need' if count != 1 else 'needs'} attention",
                            "Sorted by the existing priority ranking. Higher priority = stronger negative sentiment, "
                            "higher confidence, and more severity cue hits.",
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<div class='priority-list'>" + "".join(items) + "</div>",
                        unsafe_allow_html=True,
                    )
