# app.py
import io
import re
import unicodedata
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import html as st_html

st.set_page_config(page_title="MO Tool", layout="wide")

# -------------------- Helper rerun --------------------
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# -------------------- Global CSS --------------------
st.markdown(
    """
    <style>
    [data-testid="stDataFrame"] thead tr th,
    [data-testid="stDataFrame"] [role="columnheader"] {
        background: #eaf0ff !important;
        color: #101828 !important;
        font-weight: 800 !important;
        border-bottom: 1px solid #94a3b8 !important;
    }
    [data-testid="stDataFrame"] [role="columnheader"] * { color: #101828 !important; font-weight: 800 !important; }
    [data-testid="stDataFrame"] thead { box-shadow: 0 2px 0 rgba(0,0,0,0.06); }

    [data-testid="stDataFrame"] div[role="cell"] {
      white-space: normal !important;
      overflow-wrap: anywhere !important;
      word-break: break-word !important;
      line-height: 1.2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Session init --------------------
st.session_state.setdefault("global_df_base", None)
st.session_state.setdefault("global_df", None)
st.session_state.setdefault("ad_mapping_dict", None)
st.session_state.setdefault("ad_mapping_applied", False)
st.session_state.setdefault("filters_open", False)
st.session_state.setdefault("checkver_search", "")
st.session_state.setdefault("firebase_df", None)

# =========================
# Utils: normalize strings
# =========================
SPACE_RE = re.compile(r"\s+")
PARENS_RE = re.compile(r"\(.*?\)")
THOUSANDS_RE = re.compile(r"^\d{1,3}([.,]\d{3})+$")
DECIMAL_COMMA_RE = re.compile(r"^\d+,\d+$")
DECIMAL_DOT_RE = re.compile(r"^\d+\.\d+$")

def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def normalize_key(s: str) -> str:
    s = str(s or "").strip()
    s = PARENS_RE.sub("", s)
    s = strip_accents(s)
    s = s.replace("Đ", "D").replace("đ", "d")
    s = s.lower()
    s = SPACE_RE.sub(" ", s)
    s = s.replace("—", "-").replace("–", "-")
    return s

# =========================================
# Canonical schema + multilingual synonyms
# =========================================
CANONICAL_MAP: Dict[str, List[str]] = {
    "date": ["date", "ngay", "day", "report date"],
    "app": ["app", "application", "ung dung", "app name"],
    "app_id": ["app id", "application id", "package id", "bundle id", "id ung dung"],
    "ad_unit": ["ad unit", "ad unit name", "don vi quang cao", "adunit"],
    "ad_unit_id": ["ad unit id", "ad unit code", "id don vi quang cao", "placement id"],
    "ad_format": ["ad format", "format", "ad type", "dinh dang quang cao"],
    "country": ["country", "quoc gia"],
    "ad_source": ["ad source", "nguon quang cao", "network"],
    "platform": ["platform", "os", "nen tang"],
    "currency": ["currency", "currency code", "tien te", "ma tien te"],
    "estimated_earnings": [
        "estimated earnings", "estimated earnings usd",
        "doanh thu uoc tinh", "thu nhap uoc tinh", "thu nhap uoc tinh usd",
    ],
    "requests": ["ad requests", "requests", "yeu cau", "so yeu cau", "so luot yeu cau", "so luot yeu cau quang cao"],
    "matched_requests": ["matched requests", "matched ad requests", "so yeu cau da khop", "yeu cau da khop", "so luot yeu cau da khop"],
    "impressions": ["impressions", "ad impressions", "so luot hien thi", "so lan hien thi"],
    "clicks": ["clicks", "so luot nhap", "so lan nhap"],
    "ecpm_input": ["ecpm", "ecpm usd", "ecpm quan sat duoc", "ecpm quan sat duoc usd"],
    "rpm_input": ["rpm"],
    "version": ["version", "app version", "app_ver", "ver", "build", "build version", "release"],
}

BIDDING_BLOCKERS = ["gia thau", "gia-thau", "dau gia", "bid", "bidding", "auction"]
RATE_BLOCKERS = ["ty le", "ty-le", "rate"]
PER_VALUE_BLOCKERS = ["tren moi", "per ", "per-", "per_", "moi nguoi", "per user", "per viewer"]

# Prefix rule cho nhóm native
NATIVE_PREFIXES = [
    "native_language",
    "native_language_dup",
    "native_onboarding",
    "native_onboarding_full",
    "native_welcome",
    "native_feature",
    "native_permission",
]

# Nhóm phân tích checkver
ANALYZE_GROUPS = [
    ("inter_splash", "Interstitial Splash"),
    ("appopen_splash", "AppOpen Splash"),
    ("native_splash", "Native Splash"),
    ("native_language", "Native Language"),
    ("native_language_dup", "Native Language Dup"),
    ("native_onboarding", "Native Onboarding"),
    ("native_onboarding_full", "Native Onboarding Full"),
]

def build_reverse_map() -> Dict[str, str]:
    rev = {}
    for canon, variants in CANONICAL_MAP.items():
        for v in variants:
            rev[normalize_key(v)] = canon
    return rev

REVERSE_MAP = build_reverse_map()

# ===================================
# Readers
# ===================================
def try_read_csv(file_bytes: bytes) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    head = file_bytes[:4]
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        for sep in ["\t", ",", ";"]:
            bio.seek(0)
            try:
                df = pd.read_csv(bio, dtype=str, encoding="utf-16", sep=sep, engine="python")
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
    for enc in ["utf-8", "utf-8-sig", "cp1252"]:
        for sep in [",", ";", "\t"]:
            bio.seek(0)
            try:
                df = pd.read_csv(bio, dtype=str, encoding=enc, sep=sep, engine="python")
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    bio.seek(0)
    return pd.read_csv(bio, dtype=str, engine="python", sep=None, encoding="utf-8", encoding_errors="ignore")

def try_read_excel(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), dtype=str)

def read_any_table_from_name_bytes(name: str, b: bytes) -> pd.DataFrame:
    n = name.lower()
    if n.endswith((".xlsx", ".xls")):
        return try_read_excel(b)
    if n.endswith(".json"):
        try:
            return pd.read_json(io.BytesIO(b))
        except ValueError:
            import json
            return pd.json_normalize(json.loads(b.decode("utf-8", errors="ignore")))
    return try_read_csv(b)

# Reader đặc thù cho Firebase CSV có phần mô tả đầu file
def read_firebase_csv_bytes(b: bytes) -> pd.DataFrame:
    try:
        text = b.decode("utf-8-sig", errors="ignore")
    except Exception:
        text = b.decode("cp1252", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().lower().startswith("app version,"):
            header_idx = i
            break
    if header_idx is None:
        for i, ln in enumerate(lines):
            if "app version" in ln.lower() and "," in ln:
                header_idx = i
                break
    if header_idx is None:
        raise ValueError("Không tìm được dòng tiêu đề 'App version' trong file Firebase.")
    clean = "\n".join(lines[header_idx:])
    bio = io.StringIO(clean)
    df = pd.read_csv(bio, dtype=str, engine="python", sep=None)
    return df

# ===================================
# Header normalization and mapping
# ===================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    original_cols = list(df.columns)
    mapped = []
    used_targets = set()
    for c in original_cols:
        key = normalize_key(c)
        target = REVERSE_MAP.get(key)
        if not target:
            if "quoc gia" in key or "country" in key:
                target = "country"
            elif ("don vi quang cao" in key) or ("ad unit" in key):
                if "id" in key or key.endswith("code"):
                    target = "ad_unit_id"
                else:
                    target = "ad_unit"
            elif "dinh dang" in key or "ad format" in key or "format" in key or "ad type" in key:
                target = "ad_format"
            elif ("thu nhap" in key or "doanh thu" in key or "estimated earnings" in key):
                target = "estimated_earnings"
            elif ("yeu cau da khop" in key or ("matched" in key and "request" in key)):
                target = "matched_requests"
            elif "yeu cau" in key or "requests" in key:
                if not any(b in key for b in BIDDING_BLOCKERS):
                    target = "requests"
            elif "hien thi" in key or "impressions" in key:
                if (any(b in key for b in PER_VALUE_BLOCKERS) or any(b in key for b in RATE_BLOCKERS)):
                    target = None
                else:
                    target = "impressions"
            elif "nhap" in key or "click" in key:
                target = "clicks"
            elif "ecpm" in key:
                target = "ecpm_input"
            elif ("tien te" in key or "currency" in key):
                target = "currency"
            elif "app id" in key or ("id" in key and "app" in key):
                target = "app_id"
            elif key == "app" or "ung dung" in key:
                target = "app"
            elif key in ("os", "platform", "nen tang"):
                target = "platform"
            elif "date" in key or "ngay" in key or "report" in key:
                target = "date"
            elif "version" in key or "build" in key or "release" in key:
                target = "version"
        if target and target not in used_targets:
            mapped.append(target)
            used_targets.add(target)
        else:
            mapped.append(c)
    out = df.copy()
    out.columns = mapped
    return out

# ============================
# Number parsing and KPI math
# ============================
def to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    x = s.astype(str).fillna("").str.strip()
    x = x.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    x = x.str.replace(r"[^\d,.\-%]", "", regex=True)
    perc_mask = x.str.endswith("%")
    x = x.str.replace("%", "", regex=False)

    def parse_one(val: str) -> float:
        if val == "" or val == "-" or val == "--":
            return np.nan
        if THOUSANDS_RE.match(val):
            return float(val.replace(",", "").replace(".", ""))
        if DECIMAL_COMMA_RE.match(val):
            return float(val.replace(",", "."))
        if DECIMAL_DOT_RE.match(val):
            return float(val)
        if "," in val and "." not in val:
            parts = val.split(",")
            if all(i == 0 or len(p) == 3 for i, p in enumerate(parts)):
                return float(val.replace(",", ""))
            return float(val.replace(",", "."))
        if "." in val and "," not in val:
            parts = val.split(".")
            if all(i == 0 or len(p) == 3 for i, p in enumerate(parts)):
                return float(val.replace(".", ""))
            return float(val)
        return float(val)

    vals = x.apply(parse_one)
    vals = np.where(perc_mask, vals / 100.0, vals)
    return pd.Series(vals, index=s.index, dtype=float)

NUMERIC_COLS = ["requests", "matched_requests", "impressions", "clicks", "estimated_earnings"]

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = to_numeric_series(df[c])
    return df

# -------------------- Parse dates: nhiều trường hợp --------------------
def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    out = df.copy()
    s = out["date"]

    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if dt.notna().sum() == 0:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if dt.notna().sum() == 0:
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().any():
            vals = s_num.dropna()
            maxv = float(vals.max())
            medv = float(vals.median())
            if maxv > 1e12 or medv > 1e12:
                dt = pd.to_datetime(s_num, unit="ms", errors="coerce")
            elif maxv > 1e9 or medv > 1e9:
                dt = pd.to_datetime(s_num, unit="s", errors="coerce")
            else:
                base = pd.Timestamp("1899-12-30")
                try:
                    dt_candidate = pd.to_timedelta(s_num, unit="D") + base
                    mask_ok = (dt_candidate.dt.year >= 1990) & (dt_candidate.dt.year <= 2100)
                    dt = dt_candidate.where(mask_ok, other=pd.NaT)
                except Exception:
                    pass
    out["date"] = dt
    return out

def ensure_metric_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df

def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-12
    req = df.get("requests", 0).astype(float)
    mreq = df.get("matched_requests", 0).astype(float)
    imp = df.get("impressions", 0).astype(float)
    clk = df.get("clicks", 0).astype(float)
    rev = df.get("estimated_earnings", 0.0).astype(float)
    df["match_rate"] = np.where(req > 0, mreq / (req + eps), np.nan)
    df["show_rate_on_matched"] = np.where(mreq > 0, imp / (mreq + eps), np.nan)
    df["show_rate_on_request"] = np.where(req > 0, imp / (req + eps), np.nan)
    df["rpr"] = np.where(req > 0, rev / (req + eps), np.nan)
    df["rpm_1000req"] = df["rpr"] * 1000.0
    df["ecpm"] = np.where(imp > 0, (rev / (imp + eps)) * 1000.0, np.nan)
    df["ctr"] = np.where(imp > 0, clk / (imp + eps), np.nan)
    return df

def aggregate(df: pd.DataFrame, dims: List[str]) -> pd.DataFrame:
    metrics = ["estimated_earnings", "requests", "matched_requests", "impressions", "clicks"]
    if df is None or df.shape[0] == 0:
        if not dims:
            base = pd.DataFrame({m: [0.0] for m in metrics})
            return compute_kpis(base)
        else:
            return pd.DataFrame(columns=(dims + metrics))
    metrics_present = [m for m in metrics if m in df.columns]
    if not dims:
        sums = {m: [pd.to_numeric(df[m], errors="coerce").sum()] if m in df.columns else [0.0] for m in metrics}
        out = pd.DataFrame(sums)
        return compute_kpis(out)
    if not metrics_present:
        uniq = df[dims].drop_duplicates().reset_index(drop=True)
        for m in metrics:
            uniq[m] = 0.0
        return compute_kpis(uniq)
    grouped = df.groupby(dims, dropna=False)[metrics_present].sum().reset_index()
    for m in metrics:
        if m not in grouped.columns:
            grouped[m] = 0.0
    grouped = compute_kpis(grouped)
    return grouped.sort_values("estimated_earnings", ascending=False)

# =========================
# Loader + cache
# =========================
@st.cache_data(show_spinner=False)
def cached_prepare_any(files: List[Tuple[str, bytes]]) -> pd.DataFrame:
    frames = []
    for name, b in files:
        raw = read_any_table_from_name_bytes(name, b)
        if raw is None or raw.empty:
            continue
        raw = raw.loc[:, ~raw.columns.astype(str).str.fullmatch(r"\s*")]
        raw = raw.dropna(axis=1, how="all")
        df = normalize_columns(raw)
        df = coerce_numeric(df)
        df = parse_dates(df)
        df = ensure_metric_cols(df)
        for col in ["app", "ad_unit", "ad_format", "country", "ad_source", "platform", "currency", "version"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    df_all = compute_kpis(df_all)
    return df_all

# ================
# Mapping ad_unit -> ad_name
# ================
def pick_mapping_cols(df_map: pd.DataFrame) -> Optional[tuple]:
    code_syn = {
        "ad_unit", "ad unit", "adunit", "adunit code", "ad unit id", "ad_unit_id", "unit", "code", "ma", "id don vi quang cao", "ad id", "placement id"
    }
    name_syn = {
        "ad_name", "ad name", "adunit name", "ad unit name", "ten don vi", "ten ad", "name", "friendly name", "placement name"
    }
    code_col = None
    name_col = None
    for c in df_map.columns:
        k = normalize_key(c)
        if code_col is None and k in code_syn:
            code_col = c
        if name_col is None and k in name_syn:
            name_col = c
    if code_col and name_col:
        return (code_col, name_col)
    if df_map.shape[1] >= 2:
        return (df_map.columns[0], df_map.columns[1])
    return None

def try_read_mapping_file(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return try_read_excel(uploaded_file.getvalue())
    else:
        return try_read_csv(uploaded_file.getvalue())

def parse_mapping_text(pasted_text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not pasted_text:
        return mapping
    buf: Optional[str] = None
    for line in pasted_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep_used = None
        for sep in [",", ";", "\t", "|", ":"]:
            if sep in line:
                sep_used = sep
                break
        if sep_used:
            parts = [p.strip() for p in line.split(sep_used)]
            if len(parts) >= 2 and parts[0] and parts[1]:
                mapping[parts[0].upper()] = parts[1]
            buf = None
        else:
            if buf is None:
                buf = line
            else:
                mapping[buf.upper()] = line
                buf = None
    return mapping

def build_mapping_dict(file, pasted_text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if file is not None:
        try:
            df_map = try_read_mapping_file(file)
            if df_map is not None and not df_map.empty:
                df_map = df_map.dropna(how="all").dropna(axis=1, how="all")
                cols = pick_mapping_cols(df_map)
                if cols:
                    code_col, name_col = cols
                    for _, row in df_map.iterrows():
                        code = str(row.get(code_col, "")).strip()
                        val = str(row.get(name_col, "")).strip()
                        if code and val:
                            mapping[code.upper()] = val
        except Exception as e:
            st.warning(f"Không đọc được file mapping: {e}")
    text_map = parse_mapping_text(pasted_text)
    mapping.update(text_map)
    return mapping

# ================
# Pretty DF (formatting)
# ================
def build_pretty_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    def fmt_money(x): return "—" if pd.isna(x) else f"${x:,.2f}"
    def fmt_int(x):   return "" if pd.isna(x) else f"{int(round(x)):,.0f}"
    def fmt_pct(x):   return "" if pd.isna(x) else f"{x*100:,.2f}%"
    def fmt_rpr(x):   return "" if pd.isna(x) else f"${x:,.6f}"
    def fmt_num2(x):  return "" if pd.isna(x) else f"{x:,.2f}"
    def fmt_num6(x):  return "" if pd.isna(x) else f"{x:,.6f}"

    out = df[cols].copy()
    int_cols   = [c for c in ["requests", "matched_requests", "impressions", "clicks", "user", "new_user"] if c in out.columns]
    money_cols = [c for c in ["estimated_earnings", "ecpm", "rpm_1000req"] if c in out.columns]
    pct_cols   = [c for c in ["match_rate", "show_rate_on_matched", "show_rate_on_request", "ctr"] if c in out.columns]
    rpr_col    = [c for c in ["rpr"] if c in out.columns]
    num2_cols  = [c for c in ["imp_per_user", "imp_per_new_user", "req_per_user", "req_per_new_user"] if c in out.columns]
    num6_cols  = [c for c in ["rev_per_user", "rev_per_new_user"] if c in out.columns]

    for c in int_cols:   out[c] = out[c].apply(fmt_int)
    for c in money_cols: out[c] = out[c].apply(fmt_money)
    for c in pct_cols:   out[c] = out[c].apply(fmt_pct)
    for c in rpr_col:    out[c] = out[c].apply(fmt_rpr)
    for c in num2_cols:  out[c] = out[c].apply(fmt_num2)
    for c in num6_cols:  out[c] = out[c].apply(fmt_num6)
    return out

# ================
# Helper: ad_unit_code summary (cho Manual)
# ================
def summarize_codes(s: pd.Series) -> str:
    vals = [str(v).strip() for v in pd.unique(s.dropna()) if str(v).strip() != ""]
    if not vals: return ""
    if len(vals) == 1: return vals[0]
    short = ";".join(sorted(vals)[:3])
    return short + ("…" if len(vals) > 3 else "")

def add_adunit_code_column(agg_df: pd.DataFrame, original_df: pd.DataFrame, group_dims: List[str]) -> pd.DataFrame:
    if "ad_unit" not in original_df.columns:
        return agg_df
    code_map = (
        original_df.groupby(group_dims, dropna=False)["ad_unit"]
        .agg(summarize_codes)
        .reset_index()
        .rename(columns={"ad_unit": "ad_unit_code"})
    )
    return agg_df.merge(code_map, on=group_dims, how="left")

# ================
# Column labels/config
# ================
LABEL_MAP = {
    "app": "App",
    "version": "App version",
    "ad_unit": "Ad unit",
    "ad_name": "ad name",
    "estimated_earnings": "Est. earnings (USD)",
    "ecpm": "Observed eCPM (USD)",
    "requests": "Requests",
    "match_rate": "Match rate",
    "matched_requests": "Matched requests",
    "show_rate_on_request": "Show rate",
    "impressions": "Impressions",
    "ctr": "CTR",
    "clicks": "Clicks",
    "user": "user",
    "new_user": "new user",
    "imp_per_user": "imp/user",
    "imp_per_new_user": "imp/new user",
    "rev_per_user": "rev/user",
    "rev_per_new_user": "rev/new user",
    "req_per_user": "req/user",
    "req_per_new_user": "req/new user",
}
HELP_MAP = {
    "estimated_earnings": "Earnings: doanh thu ước tính.",
    "requests": "Ad requests.",
    "matched_requests": "Matched requests.",
    "impressions": "Impressions.",
    "clicks": "Clicks.",
    "match_rate": "matched_requests / requests",
    "show_rate_on_request": "impressions / requests",
    "ecpm": "(estimated_earnings / impressions) × 1000",
    "ctr": "clicks / impressions",
    "user": "Tổng user (Firebase) theo app + version",
    "new_user": "New user (Firebase) theo app + version",
    "imp_per_user": "impressions / user",
    "imp_per_new_user": "impressions / new_user",
    "rev_per_user": "estimated_earnings / user",
    "rev_per_new_user": "estimated_earnings / new_user",
    "req_per_user": "requests / user",
    "req_per_new_user": "requests / new user",
}

def build_column_config(cols: List[str]) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    for c in cols:
        label = LABEL_MAP.get(c, c)
        help_txt = HELP_MAP.get(c)
        if c == "ad_name":
            cfg[c] = st.column_config.TextColumn(label=label, help=help_txt, width=320)
        elif c == "ad_unit":
            cfg[c] = st.column_config.TextColumn(label=label, help=help_txt, width=160)
        elif c == "app":
            cfg[c] = st.column_config.TextColumn(label=label, help=help_txt, width=180)
        else:
            cfg[c] = st.column_config.Column(label=label, help=help_txt)
    return cfg

# =========================
# Unified table renderer + highlight
# =========================
def _wrap_for_plotly(val: str, width: int = 22) -> str:
    s = "" if val is None else str(val)
    if not s:
        return ""
    s = s.replace("_", "_<br>")
    parts = []
    for seg in s.split("<br>"):
        while len(seg) > width:
            parts.append(seg[:width])
            seg = seg[width:]
        parts.append(seg)
    return "<br>".join(parts)

def render_table(
    df_print: pd.DataFrame,
    height: int,
    cell_colors: Optional[list] = None,
    highlight_b: bool = False,
    version_b: Optional[str] = None,
):
    """
    - Nếu cell_colors is None: dùng st.dataframe (không highlight từng ô).
    - Nếu có cell_colors: dùng Plotly Table để đổ màu từng ô.
      cell_colors là ma trận [n_col][n_row] theo đúng thứ tự df_print.columns.
    """
    if cell_colors is None:
        col_cfg = build_column_config(list(df_print.columns))
        st.dataframe(df_print, use_container_width=True, hide_index=True, column_config=col_cfg, height=height)
        return

    df_wrap = df_print.copy()
    for col in [c for c in ["ad_name", "ad_unit", "app"] if c in df_wrap.columns]:
        df_wrap[col] = df_wrap[col].apply(lambda x: _wrap_for_plotly(x, width=22))

    header_color = "#eaf0ff"
    header_vals = [LABEL_MAP.get(c, c) for c in df_wrap.columns]

    col_widths = []
    for c in df_wrap.columns:
        if c == "ad_name":
            col_widths.append(320)
        elif c in ("ad_unit", "app"):
            col_widths.append(160)
        else:
            col_widths.append(110)

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=col_widths,
                header=dict(
                    values=header_vals,
                    fill_color=header_color,
                    align="left",
                    font=dict(color="#101828", size=12),
                ),
                cells=dict(
                    values=[df_wrap[c] for c in df_wrap.columns],
                    fill_color=cell_colors,
                    align="left",
                    height=30,
                ),
            )
        ]
    )
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), autosize=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def build_checkver_cell_colors(
    df_numeric: pd.DataFrame,
    print_cols: List[str],
    version_a: str,
    version_b: str,
    id_cols: List[str],
    shade_b_rows: bool = True,
) -> list:
    """
    Trả về ma trận màu [n_col][n_row] cho df_numeric theo các cột print_cols.
    - Hàng Version B: nền be nhạt (nếu shade_b_rows=True).
    - Các ô ở hàng Version B tại các cột:
        show_rate_on_request, requests,
        req_per_user, req_per_new_user,
        imp_per_user, imp_per_new_user,
        rev_per_user, rev_per_new_user
      được tô:
        + Xanh lá (#22c55e) nếu B > A
        + Cam (#fb923c) nếu B < A
      Nếu A không có giá trị: coi B>0 là tốt (xanh), B<=0 không tô.
      Cột nào NaN do rule native/non-native thì không tô.
    """
    n_rows = len(df_numeric)
    n_cols = len(print_cols)

    base_A = "white"
    base_B = "#F6EBD9" if shade_b_rows else "white"
    good = "#22c55e"
    bad = "#fb923c"

    ver = df_numeric["version"].astype(str).tolist()
    cell_colors = [[(base_B if ver[r] == str(version_b) else base_A) for r in range(n_rows)] for _ in range(n_cols)]

    def key_of_row(r: int):
        return tuple(df_numeric.loc[r, c] if c in df_numeric.columns else None for c in id_cols)

    idxA, idxB = {}, {}
    for r in range(n_rows):
        k = key_of_row(r)
        if ver[r] == str(version_a):
            idxA[k] = r
        elif ver[r] == str(version_b):
            idxB[k] = r

    comp_cols = [
        "show_rate_on_request",
        "requests",
        "req_per_user", "req_per_new_user",
        "imp_per_user", "imp_per_new_user",
        "rev_per_user", "rev_per_new_user",
    ]

    for k, rB in idxB.items():
        rA = idxA.get(k, None)
        for col in comp_cols:
            if col not in df_numeric.columns or col not in print_cols:
                continue
            cidx = print_cols.index(col)
            vB = pd.to_numeric(df_numeric.iloc[rB][col], errors="coerce")
            vA = pd.to_numeric(df_numeric.iloc[rA][col], errors="coerce") if rA is not None else np.nan
            if pd.isna(vB):
                continue
            color = None
            if pd.isna(vA):
                if pd.notna(vB) and vB > 0:
                    color = good
            else:
                if vB > vA:
                    color = good
                elif vB < vA:
                    color = bad
            if color:
                cell_colors[cidx][rB] = color

    return cell_colors

# =========================
# Excel export (auto engine)
# =========================
def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except Exception:
            raise ModuleNotFoundError("Thiếu engine Excel. Cài: pip install XlsxWriter hoặc pip install openpyxl")
    with pd.ExcelWriter(buf, engine=engine) as w:
        for name, df in sheets.items():
            df.to_excel(w, index=False, sheet_name=str(name)[:31])
    buf.seek(0)
    return buf.getvalue()

# =========================
# Keep active tab on rerun
# =========================
def persist_active_tab(tab_labels: List[str]):
    try:
        params = st.query_params
    except Exception:
        params = st.experimental_get_query_params()

    try:
        val = params.get("tab")
    except Exception:
        val = None

    if isinstance(val, list):
        desired = val[0] if val else tab_labels[0]
    elif isinstance(val, str):
        desired = val
    else:
        desired = tab_labels[0]
    if desired not in tab_labels:
        desired = tab_labels[0]

    js = f"""
    <script>
      const desired = {repr(desired)};
      const setParam = (val) => {{
        const url = new URL(window.parent.location);
        url.searchParams.set('tab', val);
        window.parent.history.replaceState(null, '', url);
      }};
      const doClick = () => {{
        const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
        for (const t of tabs) {{
          if (t.innerText.trim() === desired) {{
            if (!t.getAttribute('_restored')) {{
              t.click();
              t.setAttribute('_restored','1');
            }}
          }}
          t.addEventListener('click', () => setParam(t.innerText.trim()));
        }}
      }};
      setTimeout(doClick, 50);
    </script>
    """
    st_html(js, height=0, width=0)

# =========================
# Global data source + filters (shared)
# =========================
st.title("MO Tool")

with st.expander("Nguồn dữ liệu (dùng chung)", expanded=True):
    up = st.file_uploader(
        "Tải 1 hoặc nhiều file báo cáo (CSV/TXT/XLSX/XLS/JSON)",
        type=["csv", "txt", "xlsx", "xls", "json"],
        accept_multiple_files=True,
        key="global_upload",
    )
    if up:
        files = [(f.name, f.getvalue()) for f in up]
        df_base = cached_prepare_any(files)
        if df_base is None or df_base.empty:
            st.error("Không trích xuất được dữ liệu hợp lệ. Kiểm tra file/encoding hoặc tiêu đề cột.")
            st.session_state["global_df_base"] = None
        else:
            st.session_state["global_df_base"] = df_base
            st.success(f"Nạp {len(df_base):,} dòng dữ liệu.")

# Sidebar: Clear cache
with st.sidebar:
    if st.button("Xóa cache dữ liệu"):
        st.cache_data.clear()
        st.session_state["global_df_base"] = None
        st.session_state["global_df"] = None
        safe_rerun()

# Sidebar: Global Mapping áp dụng CHUNG
with st.sidebar.expander("Mapping Ad unit → Ad name (dùng chung)", expanded=False):
    map_file = st.file_uploader("Tệp mapping (CSV/XLSX)", type=["csv", "txt", "tsv", "xlsx", "xls"], key="global_mapfile")
    map_text = st.text_area(
        "Hoặc dán 'code,name' (hoặc 2 dòng: code xuống dòng name)",
        placeholder="I001,Banner Home\nI002,Interstitial LevelUp\n# hoặc:\nI003\nRewarded Level Complete",
        height=150,
        key="global_maptext",
    )
    current_dict = build_mapping_dict(map_file, map_text)
    if current_dict:
        preview = pd.DataFrame(list(current_dict.items()), columns=["ad_unit(code)", "ad_name"]).head(10)
        st.caption("Xem trước mapping:")
        st.dataframe(preview, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Áp dụng mapping", use_container_width=True, type="primary"):
            st.session_state["ad_mapping_dict"] = current_dict if current_dict else None
            st.session_state["ad_mapping_applied"] = bool(current_dict)
            safe_rerun()
    with c2:
        if st.button("Gỡ mapping", use_container_width=True):
            st.session_state["ad_mapping_dict"] = None
            st.session_state["ad_mapping_applied"] = False
            safe_rerun()

# Chuẩn bị global_df (mapping + filters)
df_work = None
if isinstance(st.session_state.get("global_df_base"), pd.DataFrame) and not st.session_state["global_df_base"].empty:
    df_work = st.session_state["global_df_base"].copy()
    if st.session_state["ad_mapping_applied"] and st.session_state["ad_mapping_dict"] and "ad_unit" in df_work.columns:
        def map_code(v):
            if v is None:
                return np.nan
            return st.session_state["ad_mapping_dict"].get(str(v).strip().upper(), np.nan)
        df_work["ad_name"] = df_work["ad_unit"].apply(map_code)
    else:
        if "ad_name" in df_work.columns:
            df_work = df_work.drop(columns=["ad_name"])
else:
    df_work = None

# Sidebar: GLOBAL filters
def apply_global_filters(dframe: pd.DataFrame) -> pd.DataFrame:
    if dframe is None or dframe.empty:
        return dframe
    st.sidebar.checkbox("Mở bộ lọc", value=st.session_state["filters_open"], key="filters_open")
    with st.sidebar.expander("Bộ lọc chung", expanded=st.session_state["filters_open"]):
        df = dframe.copy()
        if "date" in df.columns and df["date"].notna().any():
            min_date = pd.to_datetime(df["date"]).min()
            max_date = pd.to_datetime(df["date"]).max()
            dr = st.date_input("Khoảng ngày", value=(min_date.date(), max_date.date()))
            if isinstance(dr, tuple) and len(dr) == 2:
                start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
                df = df[(df["date"] >= start) & (df["date"] <= end)]

        def multiselect_filter(label, col, dframe):
            if col in dframe.columns:
                opts = sorted([x for x in dframe[col].dropna().unique() if str(x).strip() != ""])
                selected = st.multiselect(label, options=opts, default=[])
                if selected:
                    return dframe[dframe[col].isin(selected)]
            return dframe

        df = multiselect_filter("Currency", "currency", df)
        df = multiselect_filter("App", "app", df)
        if "ad_name" in df.columns:
            df = multiselect_filter("Ad name", "ad_name", df)
        df = multiselect_filter("Ad format", "ad_format", df)
        df = multiselect_filter("Ad unit (code)", "ad_unit", df)
        df = multiselect_filter("Country", "country", df)
        df = multiselect_filter("Ad source", "ad_source", df)
        df = multiselect_filter("Platform", "platform", df)
        if "version" in df.columns:
            df = multiselect_filter("Version (lọc trước khi so sánh)", "version", df)
    return df

if df_work is not None:
    st.session_state["global_df"] = apply_global_filters(df_work)
else:
    st.session_state["global_df"] = None

# Sidebar: kích thước
with st.sidebar.expander("Kích thước bảng", expanded=False):
    max_width = st.slider("Độ rộng tối đa trang (px)", min_value=1200, max_value=2200, value=1700, step=50)
    table_height = st.slider("Chiều cao bảng (px)", min_value=400, max_value=1200, value=800, step=20)
st.markdown(f"<style>.block-container{{max-width:{max_width}px !important;}}</style>", unsafe_allow_html=True)

# =========================
# Tabs + persist active
# =========================
tab_labels = ["Manual Floor Log", "Checkver"]
tabs = st.tabs(tab_labels)
persist_active_tab(tab_labels)

# =========================
# TAB 1: Manual Floor Log
# =========================
with tabs[0]:
    st.subheader("Bảng tổng hợp")

    with st.expander("Hướng dẫn", expanded=True):
        st.markdown(
            """
- B1: Upload file csv lên
- B2: Ở phần bộ lọc phía bên trái thì có thể chọn theo tùy chọn, nếu chưa có mapping ads unit code thì vào phần ads unit code làm theo hướng dẫn
- B3: Bật option mapping ads unit code để report xem dễ hơn
- B4: Nếu muốn xem nhiều report thì add thêm, còn không thì chỉ việc X file csv đó đi thì sẽ không bị lẫn data
            """
        )

    df = st.session_state.get("global_df")
    if df is None or df.empty:
        st.info("Chưa có dữ liệu sau bộ lọc chung.")
    else:
        def fmt_money(x):
            try:
                return f"${x:,.2f}"
            except Exception:
                return "—"
        def fmt_pct(x): return f"{x*100:,.2f}%" if pd.notnull(x) else "—"
        def fmt_float(x): return f"{x:,.6f}" if pd.notnull(x) else "—"

        total = aggregate(df, [])
        tot = total.iloc[0]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Earnings", fmt_money(tot.get("estimated_earnings", 0.0)))
        c2.metric("Requests", f"{int(tot.get('requests', 0)):,}")
        c3.metric("Impressions", f"{int(tot.get('impressions', 0)):,}")
        c4.metric("Match rate", fmt_pct(tot.get("match_rate")))
        c5.metric("Show rate (matched)", fmt_pct(tot.get("show_rate_on_matched")))
        c6.metric("eCPM", fmt_money(tot.get("ecpm", 0.0)))
        c7, c8, c9 = st.columns(3)
        c7.metric("RPR (Rev/Request)", fmt_float(tot.get("rpr")))
        c8.metric("RPM (per 1000 req)", fmt_money(tot.get("rpm_1000req", 0.0)))
        c9.metric("CTR", fmt_pct(tot.get("ctr")))

        dims_all = ["date", "app", "ad_name", "ad_format", "ad_unit", "country", "ad_source", "platform", "currency", "version"]
        dims_available = [d for d in dims_all if d in df.columns]
        default_dims = [d for d in dims_available if d not in ("date", "currency")][:2]
        group_dims = st.multiselect("Nhóm theo (tối đa 3)", options=dims_available, default=default_dims, max_selections=3)

        show_code = False
        if "ad_name" in df.columns and ("ad_name" in group_dims) and ("ad_unit" not in group_dims):
            show_code = st.checkbox("Thêm cột ad_unit (code) cạnh ad_name", value=False, key="manual_show_code")

        agg_df = aggregate(df, group_dims)
        if show_code:
            agg_df = add_adunit_code_column(agg_df, df, group_dims)

        display_cols = group_dims + [
            "estimated_earnings", "requests", "matched_requests", "impressions", "clicks",
            "match_rate", "show_rate_on_matched", "show_rate_on_request", "rpr", "rpm_1000req", "ecpm", "ctr",
        ]
        if show_code and "ad_unit_code" in agg_df.columns and "ad_name" in group_dims:
            idx = group_dims.index("ad_name")
            insert_pos = idx + 1
            display_cols = group_dims[:insert_pos] + ["ad_unit_code"] + group_dims[insert_pos:] + [
                "estimated_earnings", "requests", "matched_requests", "impressions", "clicks",
                "match_rate", "show_rate_on_matched", "show_rate_on_request", "rpr", "rpm_1000req", "ecpm", "ctr",
            ]
        display_cols = [c for c in display_cols if c in agg_df.columns]

        show_pretty = st.checkbox("Hiển thị số đẹp (%, tiền, dấu phẩy)", value=True, key="pretty_manual")
        show_stt = st.checkbox("Hiển thị STT", value=False, key="stt_manual")

        df_display = build_pretty_df(agg_df, display_cols) if show_pretty else agg_df[display_cols].copy()
        df_display = df_display.reset_index(drop=True)
        if show_stt:
            df_display.insert(0, "STT", np.arange(1, len(df_display) + 1))

        col_cfg = build_column_config(list(df_display.columns))
        st.dataframe(df_display, use_container_width=True, hide_index=True, column_config=col_cfg, height=table_height)

        csv_bytes = agg_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Tải CSV kết quả", data=csv_bytes, file_name="admob_aggregated.csv", mime="text/csv")

        st.subheader("Biểu đồ")
        valid_dates = 0
        if "date" in df.columns:
            try:
                valid_dates = pd.to_datetime(df["date"], errors="coerce").notna().sum()
            except Exception:
                valid_dates = df["date"].notna().sum()
        st.caption(f"Số dòng có 'date' hợp lệ sau lọc: {int(valid_dates) if 'date' in df.columns else 0}")

        if "date" in df.columns and valid_dates > 0:
            ts = aggregate(df, ["date"]).sort_values("date")
            st.plotly_chart(px.line(ts, x="date", y="estimated_earnings", title="Earnings theo ngày"), use_container_width=True)
            metric_options = ["estimated_earnings", "requests", "impressions", "rpr", "ecpm", "match_rate", "show_rate_on_matched", "ctr"]
            metric_pick = st.selectbox("Chọn metric", metric_options, index=0, key="chart_metric")
            colorable = [d for d in ["ad_name", "ad_unit", "ad_format", "country", "app", "ad_source", "platform", "currency", "version"] if d in df.columns]
            color_dim = st.selectbox("Phân rã theo", ["(none)"] + colorable, index=0, key="chart_color")
            color_dim = None if color_dim == "(none)" else color_dim
            dims = ["date"] + ([color_dim] if color_dim else [])
            ts2 = aggregate(df, dims).sort_values("date")
            title2 = f"{metric_pick} theo ngày" + (f" by {color_dim}" if color_dim else "")
            st.plotly_chart(px.line(ts2, x="date", y=metric_pick, color=color_dim, title=title2), use_container_width=True)
        else:
            st.info("Không vẽ được vì cột 'date' không hợp lệ hoặc sau lọc không còn dữ liệu ngày.")

# =========================
# Helpers cho Firebase (users/new users)
# =========================
def normalize_fb_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        k = normalize_key(c)
        if k in {"app", "ung dung", "application", "app name"} and "app" not in col_map:
            col_map["app"] = c
        elif ("version" in k or "build" in k or "release" in k) and "version" not in col_map:
            col_map["version"] = c
        elif any(x in k for x in ["new user", "newuser", "first_open", "first open", "new users"]) and "new_user" not in col_map:
            col_map["new_user"] = c
        elif any(x in k for x in ["active user", "active users", "user", "users", "dau"]) and "user" not in col_map:
            col_map["user"] = c
    out = df.copy()
    out = out.rename(columns={v: k for k, v in col_map.items()})
    keep = [c for c in ["app", "version", "user", "new_user"] if c in out.columns]
    out = out[keep]
    for c in ["user", "new_user"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["app", "version"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    return out

def load_firebase_df(uploaded_files) -> Optional[pd.DataFrame]:
    if not uploaded_files:
        return None
    frames = []
    for f in uploaded_files:
        name = getattr(f, "name", "").lower()
        try:
            if name.endswith((".csv", ".txt", ".tsv")):
                df_raw = read_firebase_csv_bytes(f.getvalue())
            elif name.endswith((".xlsx", ".xls", ".json")):
                df_raw = read_any_table_from_name_bytes(f.name, f.getvalue())
            else:
                df_raw = read_firebase_csv_bytes(f.getvalue())
            if df_raw is None or df_raw.empty:
                continue
            df_norm = normalize_fb_columns(df_raw)
            if not df_norm.empty:
                frames.append(df_norm)
        except Exception as e:
            st.warning(f"Lỗi đọc Firebase file {getattr(f,'name','')}: {e}")
    if not frames:
        return None
    fb = pd.concat(frames, ignore_index=True)
    keys = [c for c in ["app", "version"] if c in fb.columns]
    if not keys:
        return None
    for c in ["user", "new_user"]:
        if c not in fb.columns:
            fb[c] = np.nan
    agg = fb.groupby(keys, dropna=False)[["user", "new_user"]].sum(min_count=1).reset_index()
    for c in keys:
        agg[c] = agg[c].astype(str).str.strip()
    return agg

# =========================
# TAB 2: Checkver
# =========================
def detect_version_col(df: pd.DataFrame) -> Optional[str]:
    cands = []
    for c in df.columns:
        lc = c.lower().strip()
        if lc in {"version", "app ver", "app_ver", "ver", "build", "build version", "release"}:
            return c
        if "version" in lc or "build" in lc or "release" in lc:
            cands.append(c)
    return cands[0] if cands else None

def detect_key_col(df: pd.DataFrame) -> str:
    cands = []
    for c in df.columns:
        lc = c.lower()
        if lc in {"ad_unit_id", "ad_unit"} or ("ad" in lc and "unit" in lc):
            cands.append(c)
    ordered = sorted(set(cands), key=lambda x: (x.lower() not in {"ad_unit_id", "ad_unit"}, x.lower()))
    return ordered[0] if ordered else df.columns[0]

def version_tuple(v: str) -> Tuple[int, ...]:
    v = str(v or "")
    parts = re.split(r"[^\d]+", v)
    nums = tuple(int(p) for p in parts if p.isdigit())
    return nums if nums else (0,)

def apply_native_rev_rule(df: pd.DataFrame, mapping_applied: bool) -> pd.DataFrame:
    if df is None or df.empty or not mapping_applied:
        return df
    df = df.copy()
    prefixes = tuple(NATIVE_PREFIXES)
    mask_native = False
    if "ad_name" in df.columns:
        mask_native = df["ad_name"].astype(str).str.lower().str.startswith(prefixes)
    if "ad_unit" in df.columns:
        mask_native = mask_native | df["ad_unit"].astype(str).str.lower().str.startswith(prefixes)
    mask_other = ~mask_native
    for col in ["imp_per_user", "req_per_user", "rev_per_user"]:
        if col in df.columns:
            df.loc[mask_native, col] = np.nan
    for col in ["imp_per_new_user", "req_per_new_user", "rev_per_new_user"]:
        if col in df.columns:
            df.loc[mask_other, col] = np.nan
    return df

def format_pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x*100:,.2f}%"

def format_num2(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:,.2f}"

def format_num6(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:,.6f}"

def analyze_group(agg_df: pd.DataFrame, prefix: str, version_a: str, version_b: str) -> Optional[Dict[str, Dict[str, float]]]:
    if agg_df is None or agg_df.empty:
        return None
    def filt(d: pd.DataFrame, ver: str) -> pd.DataFrame:
        d = d[d["version"].astype(str) == str(ver)]
        mask = d["ad_unit"].astype(str).str.lower().str.startswith(prefix)
        if "ad_name" in d.columns:
            mask = mask | d["ad_name"].astype(str).str.lower().str.startswith(prefix)
        return d[mask].copy()
    A = filt(agg_df, version_a)
    B = filt(agg_df, version_b)
    if B.empty:
        return None
    def enrich_and_summarize(d: pd.DataFrame) -> Dict[str, float]:
        if d is None or d.empty:
            return dict(match_rate=np.nan, show_rate=np.nan, ctr=np.nan,
                        imp_user_sum=np.nan, imp_new_user_sum=np.nan,
                        req_user_sum=np.nan, req_new_user_sum=np.nan,
                        rev_user_sum=np.nan, rev_new_user_sum=np.nan)
        req = pd.to_numeric(d["requests"], errors="coerce").sum()
        mreq = pd.to_numeric(d["matched_requests"], errors="coerce").sum()
        imp = pd.to_numeric(d["impressions"], errors="coerce").sum()
        clk = pd.to_numeric(d["clicks"], errors="coerce").sum()
        rev = pd.to_numeric(d["estimated_earnings"], errors="coerce").sum()
        eps = 1e-12
        match_rate = (mreq / (req + eps)) if req > 0 else np.nan
        show_rate = (imp / (req + eps)) if req > 0 else np.nan
        ctr = (clk / (imp + eps)) if imp > 0 else np.nan
        user = pd.to_numeric(d.get("user", np.nan), errors="coerce")
        new_user = pd.to_numeric(d.get("new_user", np.nan), errors="coerce")
        imp_user = np.where((user > 0), pd.to_numeric(d["impressions"], errors="coerce") / (user + eps), np.nan)
        imp_new = np.where((new_user > 0), pd.to_numeric(d["impressions"], errors="coerce") / (new_user + eps), np.nan)
        req_user = np.where((user > 0), pd.to_numeric(d["requests"], errors="coerce") / (user + eps), np.nan)
        req_new = np.where((new_user > 0), pd.to_numeric(d["requests"], errors="coerce") / (new_user + eps), np.nan)
        rev_user = np.where((user > 0), pd.to_numeric(d["estimated_earnings"], errors="coerce") / (user + eps), np.nan)
        rev_new = np.where((new_user > 0), pd.to_numeric(d["estimated_earnings"], errors="coerce") / (new_user + eps), np.nan)
        out = dict(
            match_rate=match_rate,
            show_rate=show_rate,
            ctr=ctr,
            imp_user_sum=np.nansum(imp_user),
            imp_new_user_sum=np.nansum(imp_new),
            req_user_sum=np.nansum(req_user),
            req_new_user_sum=np.nansum(req_new),
            rev_user_sum=np.nansum(rev_user),
            rev_new_user_sum=np.nansum(rev_new),
        )
        return out
    return {"A": enrich_and_summarize(A), "B": enrich_and_summarize(B)}

def analysis_to_text(prefix: str, label: str, data: Dict[str, Dict[str, float]], version_a: str, version_b: str) -> List[str]:
    out = []
    A = data["A"]; B = data["B"]
    native = prefix.startswith("native_")
    pairs = [
        ("match rate", "match_rate", "pct"),
        ("show rate", "show_rate", "pct"),
        ("CTR", "ctr", "pct"),
    ]
    if native:
        pairs += [
            ("imp/new user", "imp_new_user_sum", "num2"),
            ("request/new user", "req_new_user_sum", "num2"),
            ("rev/new user", "rev_new_user_sum", "num6"),
        ]
    else:
        pairs += [
            ("imp/user", "imp_user_sum", "num2"),
            ("request/user", "req_user_sum", "num2"),
            ("rev/user", "rev_user_sum", "num6"),
        ]
    def fmt(v, kind):
        if kind == "pct":   return format_pct(v)
        if kind == "num2":  return format_num2(v)
        if kind == "num6":  return format_num6(v)
        return str(v)
    for title, key, kind in pairs:
        va = A.get(key, np.nan)
        vb = B.get(key, np.nan)
        if pd.isna(vb) and pd.isna(va):
            continue
        trend = "tốt hơn" if (pd.notna(vb) and pd.notna(va) and vb > va) else ("kém hơn" if (pd.notna(vb) and pd.notna(va) and vb < va) else "bằng")
        out.append(f"- {label}: {title} {version_b} = {fmt(vb, kind)}, {trend} {version_a} = {fmt(va, kind)}.")
    return out

with tabs[1]:
    st.subheader("Checkver — So sánh 2 version (1 tệp)")

    with st.expander("Hướng dẫn", expanded=True):
        st.markdown(
            """
- B1: Upload file CSV Admob lên, CVS này có thể dùng chung với Manual Floor Log
- B2: Upload file CSV từ Firebase Analytics phần Version. Upload xong nhấn nạp firebase 
- B3: Phần bộ lọc phía bên trái nếu chưa mapping thì phải thêm mapping ads name vào không là lỗi, không ra đâu
- B4: Chọn version. Version A là ver cũ, Version B là ver mới
- B5: Ở cuối cùng có nút Checkver. Ấn vào sẽ ra checkver
            """
        )

    df_all = st.session_state.get("global_df")

    with st.expander("Dữ liệu Firebase (users/new users) — tuỳ chọn", expanded=False):
        st.caption("Nạp file Firebase (CSV/XLSX/TXT/JSON) có cột App version và số liệu Active users, New users.")
        fb_files = st.file_uploader(
            "Upload Firebase files", type=["csv", "txt", "tsv", "xlsx", "xls", "json"], accept_multiple_files=True, key="fb_upload"
        )
        if st.button("Nạp Firebase", use_container_width=False):
            st.session_state["firebase_df"] = load_firebase_df(fb_files)
            if st.session_state["firebase_df"] is None or st.session_state["firebase_df"].empty:
                st.warning("Chưa lấy được dữ liệu Firebase hợp lệ.")
            else:
                st.success(f"Đã nạp {len(st.session_state['firebase_df']):,} dòng Firebase (App+Version).")
                st.dataframe(st.session_state["firebase_df"].head(10), use_container_width=True, hide_index=True)

    if df_all is None or df_all.empty:
        st.info("Chưa có dữ liệu sau bộ lọc chung.")
    else:
        ver_col_auto = detect_version_col(df_all) or ("version" if "version" in df_all.columns else df_all.columns[0])
        if any(c in df_all.columns for c in ["ad_unit", "ad_unit_id"]):
            key_col_auto = detect_key_col(df_all)
        elif "ad_name" in df_all.columns:
            key_col_auto = "ad_name"
        else:
            key_col_auto = df_all.columns[0]
        ver_col = ver_col_auto
        key_col = key_col_auto

        with st.expander("Tùy chọn nâng cao (chỉ mở nếu phát hiện sai)", expanded=False):
            ver_col = st.selectbox(
                "Cột Version",
                options=list(df_all.columns),
                index=list(df_all.columns).index(ver_col),
                help="Mặc định đã tự phát hiện. Chỉ đổi nếu nhận sai."
            )
            compare_choice = st.radio(
                "So sánh theo",
                options=["Tự động", "Ad unit", "Ad name"],
                index=0,
                horizontal=True,
                help="Tự động: ưu tiên ad_unit/ad_unit_id; nếu không có thì dùng ad_name."
            )
            if compare_choice == "Ad unit":
                ad_candidates = [c for c in df_all.columns if ("ad" in c.lower() and "unit" in c.lower())] or list(df_all.columns)
                key_col = st.selectbox("Cột Ad unit", options=ad_candidates,
                                       index=ad_candidates.index(key_col) if key_col in ad_candidates else 0)
            elif compare_choice == "Ad name":
                if "ad_name" in df_all.columns:
                    key_col = "ad_name"
                    st.caption("Đang so sánh theo: ad_name")
                else:
                    key_col = st.selectbox("Chọn cột tên quảng cáo (không có ad_name)", options=list(df_all.columns))

        versions = [str(v) for v in sorted(df_all[ver_col].dropna().astype(str).unique(), key=version_tuple)]
        if len(versions) < 2:
            st.warning("Cột Version cần có ít nhất 2 giá trị khác nhau để so sánh.")
        else:
            colv1, colv2 = st.columns(2)
            idx_latest = len(versions) - 1
            version_a = colv1.selectbox("Version A", options=versions, index=max(0, idx_latest - 1))
            version_b = colv2.selectbox("Version B (highlight)", options=versions, index=idx_latest)

            base_dims = ["app", ver_col, key_col]
            if key_col != "ad_name" and "ad_name" in df_all.columns:
                base_dims.append("ad_name")
            group_dims = [c for c in base_dims if c in df_all.columns]
            agg = aggregate(df_all, group_dims)

            rename_cols = {}
            if ver_col != "version": rename_cols[ver_col] = "version"
            if key_col != "ad_unit" and key_col in agg.columns and key_col != "ad_name":
                rename_cols[key_col] = "ad_unit"
            agg = agg.rename(columns=rename_cols)
            if key_col == "ad_name" and "ad_unit" not in agg.columns:
                agg["ad_unit"] = agg["ad_name"]

            setA = set(agg.loc[agg["version"].astype(str) == str(version_a), "ad_unit"])
            setB = set(agg.loc[agg["version"].astype(str) == str(version_b), "ad_unit"])
            common_keys = sorted(setA & setB)

            view = st.radio(
                "Hiển thị",
                options=["Common (có ở cả 2)", f"Only in {version_a}", f"Only in {version_b}", "All (2 phiên bản đã chọn)"],
                horizontal=True,
            )

            c_search, c_clear = st.columns([0.94, 0.06])
            with c_search:
                st.session_state["checkver_search"] = st.text_input(
                    "Lọc theo từ khoá (trên Ad unit/Ad name)", value=st.session_state.get("checkver_search", "")
                )
            with c_clear:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                if st.button("✕", help="Xoá bộ lọc từ khoá"):
                    st.session_state["checkver_search"] = ""
                    safe_rerun()
            search = st.session_state["checkver_search"].strip().lower()

            shade_b = st.checkbox("Tô màu Version B", value=True)

            if view.startswith("Common"):
                df_view = agg[agg["ad_unit"].isin(common_keys) & agg["version"].isin([version_a, version_b])].copy()
            elif view.startswith("Only in") and version_a in view:
                onlyA = sorted(setA - setB)
                df_view = agg[(agg["version"] == version_a) & (agg["ad_unit"].isin(onlyA))].copy()
            elif view.startswith("Only in") and version_b in view:
                onlyB = sorted(setB - setA)
                df_view = agg[(agg["version"] == version_b) & (agg["ad_unit"].isin(onlyB))].copy()
            else:
                df_view = agg[agg["version"].isin([version_a, version_b])].copy()

            if search:
                cols_s = [c for c in ["ad_unit", "ad_name"] if c in df_view.columns]
                if cols_s:
                    mask = False
                    for c in cols_s:
                        mask = mask | df_view[c].astype(str).str.lower().str.contains(search)
                    df_view = df_view[mask]

            if "ad_name" in df_view.columns:
                df_view["_sort_name"] = df_view["ad_name"].astype(str)
            else:
                df_view["_sort_name"] = df_view["ad_unit"].astype(str)
            df_view["_ver_tuple"] = df_view["version"].map(version_tuple)
            sort_cols = [c for c in ["app"] if c in df_view.columns] + ["_sort_name", "_ver_tuple"]
            df_view = df_view.sort_values(sort_cols, kind="stable").drop(columns=["_sort_name", "_ver_tuple"])

            fb_df: Optional[pd.DataFrame] = st.session_state.get("firebase_df")
            if isinstance(fb_df, pd.DataFrame) and not fb_df.empty:
                join_keys = [k for k in ["app", "version"] if k in df_view.columns and k in fb_df.columns]
                if join_keys:
                    df_view = df_view.merge(fb_df, on=join_keys, how="left")
            for c in ["user", "new_user"]:
                if c not in df_view.columns:
                    df_view[c] = np.nan

            user = pd.to_numeric(df_view["user"], errors="coerce")
            new_user = pd.to_numeric(df_view["new_user"], errors="coerce")
            eps = 1e-12
            df_view["imp_per_user"] = np.where(user > 0, df_view["impressions"].astype(float) / (user + eps), np.nan)
            df_view["imp_per_new_user"] = np.where(new_user > 0, df_view["impressions"].astype(float) / (new_user + eps), np.nan)
            df_view["rev_per_user"] = np.where(user > 0, df_view["estimated_earnings"].astype(float) / (user + eps), np.nan)
            df_view["rev_per_new_user"] = np.where(new_user > 0, df_view["estimated_earnings"].astype(float) / (new_user + eps), np.nan)
            df_view["req_per_user"] = np.where(user > 0, df_view["requests"].astype(float) / (user + eps), np.nan)
            df_view["req_per_new_user"] = np.where(new_user > 0, df_view["requests"].astype(float) / (new_user + eps), np.nan)

            df_view = apply_native_rev_rule(df_view, mapping_applied=bool(st.session_state.get("ad_mapping_applied")))

            ordered_cols = [
                "app", "version", "ad_unit", "ad_name",
                "estimated_earnings", "ecpm", "requests", "match_rate", "matched_requests",
                "show_rate_on_request", "impressions", "ctr", "clicks",
                "user", "new_user",
                "imp_per_user", "imp_per_new_user",
                "rev_per_user", "rev_per_new_user",
                "req_per_user", "req_per_new_user",
            ]
            for c in ordered_cols:
                if c not in df_view.columns:
                    df_view[c] = np.nan
            df_view = df_view[ordered_cols]

            show_pretty = st.checkbox("Hiển thị số đẹp (%, tiền, dấu phẩy)", value=True, key="pretty_checkver")
            show_stt = st.checkbox("Hiển thị STT", value=False, key="stt_checkver")
            df_print = build_pretty_df(df_view, ordered_cols) if show_pretty else df_view[ordered_cols].copy()
            df_print = df_print.reset_index(drop=True)
            if show_stt:
                df_print.insert(0, "STT", np.arange(1, len(df_print) + 1))

            # ---------- Hiển thị bảng với highlight từng ô ----------
            id_cols = [c for c in ["app", "ad_unit"] if c in df_view.columns]
            if not id_cols:
                id_cols = [c for c in ["app", "ad_name"] if c in df_view.columns]
            if not id_cols:
                id_cols = ["ad_unit"] if "ad_unit" in df_view.columns else ["ad_name"]
            print_cols = list(df_print.columns)
            cell_colors = build_checkver_cell_colors(
                df_numeric=df_view.reset_index(drop=True),
                print_cols=print_cols,
                version_a=version_a,
                version_b=version_b,
                id_cols=id_cols,
                shade_b_rows=bool(shade_b),
            )
            render_table(df_print, table_height, cell_colors=cell_colors)

            # ---------- Export ----------
            xls = to_excel_bytes({"Checkver": df_view[ordered_cols]})
            st.download_button(
                "Tải Excel (theo bảng hiện tại)",
                data=xls,
                file_name=f"checkver_{version_a}_vs_{version_b}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # ---------- PHÂN TÍCH CHECKVER ----------
            st.markdown("---")
            st.subheader("Phân tích checkver")
            if st.button("Phân tích checkver", type="primary"):
                st.write(f"So sánh {version_b} vs {version_a} theo danh mục: {', '.join([g[0] for g in ANALYZE_GROUPS])}")
                base = agg.copy()
                fb_df2 = st.session_state.get("firebase_df")
                if isinstance(fb_df2, pd.DataFrame) and not fb_df2.empty:
                    join_keys = [k for k in ["app", "version"] if k in base.columns and k in fb_df2.columns]
                    if join_keys:
                        base = base.merge(fb_df2, on=join_keys, how="left")
                else:
                    if "user" not in base.columns: base["user"] = np.nan
                    if "new_user" not in base.columns: base["new_user"] = np.nan
                if "ad_unit" not in base.columns and "ad_name" in base.columns:
                    base["ad_unit"] = base["ad_name"]

                results_any = False
                for prefix, label in ANALYZE_GROUPS:
                    data = analyze_group(base, prefix, version_a, version_b)
                    if data is None:
                        continue
                    results_any = True
                    lines = analysis_to_text(prefix, label, data, version_a, version_b)
                    if lines:
                        st.markdown(f"- Nhóm: {label} (prefix: {prefix})")
                        for ln in lines:
                            st.write(ln)
                        st.write("")
                if not results_any:
                    st.info("Version B không có ads thuộc các nhóm trong danh sách phân tích.")