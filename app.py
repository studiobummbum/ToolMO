# app.py
import io
import re
import unicodedata
from typing import List, Dict, Optional, Tuple, Set
from datetime import timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

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
st.session_state.setdefault("checkver_daycols", {})  # { "verA__verB": ["YYYY-MM-DD", ...] }

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

# Reader đặc thù cho Firebase CSV
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
            elif ("dinh dang" in key) or ("ad format" in key) or ("format" in key) or ("ad type" in key):
                target = "ad_format"
            elif ("thu nhap" in key) or ("doanh thu" in key) or ("estimated earnings" in key):
                target = "estimated_earnings"
            elif ("yeu cau da khop" in key) or (("matched" in key) and ("request" in key)):
                target = "matched_requests"
            elif ("yeu cau" in key) or ("requests" in key):
                if not any(b in key for b in BIDDING_BLOCKERS):
                    target = "requests"
            elif ("hien thi" in key) or ("impressions" in key):
                if (any(b in key for b in PER_VALUE_BLOCKERS) or any(b in key for b in RATE_BLOCKERS)):
                    target = None
                else:
                    target = "impressions"
            elif ("nhap" in key) or ("click" in key):
                target = "clicks"
            elif "ecpm" in key:
                target = "ecpm_input"
            elif ("tien te" in key) or ("currency" in key):
                target = "currency"
            elif ("app id" in key) or (("id" in key) and ("app" in key)):
                target = "app_id"
            elif (key == "app") or ("ung dung" in key):
                target = "app"
            elif key in ("os", "platform", "nen tang"):
                target = "platform"
            elif ("date" in key) or ("ngay" in key) or ("report" in key):
                target = "date"
            elif ("version" in key) or ("build" in key) or ("release" in key):
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
        if val in ("", "-", "--"):
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

# -------------------- Parse dates --------------------
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

# KHÔNG dùng eps
def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    req = df.get("requests", 0).astype(float)
    mreq = df.get("matched_requests", 0).astype(float)
    imp = df.get("impressions", 0).astype(float)
    clk = df.get("clicks", 0).astype(float)
    rev = df.get("estimated_earnings", 0.0).astype(float)

    df["match_rate"] = mreq.divide(req).where(req > 0)
    df["show_rate_on_matched"] = imp.divide(mreq).where(mreq > 0)
    df["show_rate_on_request"] = imp.divide(req).where(req > 0)
    df["rpr"] = rev.divide(req).where(req > 0)
    df["rpm_1000req"] = df["rpr"] * 1000.0
    df["ecpm"] = (rev.divide(imp).where(imp > 0)) * 1000.0
    df["ctr"] = clk.divide(imp).where(imp > 0)
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
# Loader + cache (khử trùng lặp)
# =========================
@st.cache_data(show_spinner=False)
def cached_prepare_any(files: List[Tuple[str, bytes]]) -> pd.DataFrame:
    frames = []
    for name, b in files:
        raw = read_any_table_from_name_bytes(name, b)
        if raw is None or raw.empty:
            continue
        raw = raw.loc[:, ~raw.columns.astype(str).str.fullmatch(r"\s*")]
        raw = raw.dropna(axis=1, how="all").dropna(how="all")
        df = normalize_columns(raw)
        df = coerce_numeric(df)
        df = parse_dates(df)
        df = ensure_metric_cols(df)
        for col in ["app", "ad_unit", "ad_format", "country", "ad_source", "platform", "currency", "version"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()
        df = df.drop_duplicates(ignore_index=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    before = len(df_all)
    df_all = df_all.drop_duplicates(ignore_index=True)
    dropped = before - len(df_all)
    df_all = compute_kpis(df_all)
    df_all.attrs["dupe_dropped"] = int(dropped)
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
):
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
    allowed_keys: Optional[Set[Tuple]] = None,
) -> list:
    n_rows = len(df_numeric)
    n_cols = len(print_cols)

    base_A = "white"
    base_B = "#F6EBD9" if shade_b_rows else "white"
    good = "#22c55e"
    bad = "#fb923c"

    ver = df_numeric["version"].astype(str).tolist()

    def key_of_row(r: int) -> Tuple:
        return tuple(df_numeric.loc[r, c] if c in df_numeric.columns else None for c in id_cols)

    allowed_row = []
    for r in range(n_rows):
        if allowed_keys is None:
            allowed_row.append(True)
        else:
            allowed_row.append(key_of_row(r) in allowed_keys)

    cell_colors = [[
        (base_B if (ver[r] == str(version_b) and allowed_row[r]) else base_A)
        for r in range(n_rows)
    ] for _ in range(n_cols)]

    idxA, idxB = {}, {}
    for r in range(n_rows):
        k = key_of_row(r)
        if ver[r] == str(version_a):
            idxA[k] = r
        elif ver[r] == str(version_b):
            idxB[k] = r

    comp_cols = [
        "show_rate_on_request",
        "req_per_user", "req_per_new_user",
        "imp_per_user", "imp_per_new_user",
        "rev_per_user", "rev_per_new_user",
    ]

    for k, rB in idxB.items():
        rA = idxA.get(k, None)
        if allowed_keys is not None and k not in allowed_keys:
            continue
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
            dropped = int(df_base.attrs.get("dupe_dropped", 0))
            if dropped > 0:
                st.info(f"Đã khử trùng lặp {dropped:,} dòng giống hệt nhau giữa các file upload để tránh cộng trùng số liệu.")

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
- B2: Chọn lọc ở sidebar; nếu chưa có mapping ads unit code thì thêm ở phần mapping
- B3: Bật mapping để report dễ đọc
- B4: Nếu xem nhiều report thì upload thêm; muốn xoá dữ liệu cũ thì clear cache hoặc bỏ file cũ
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
# TAB 2: Checkver — So sánh phiên bản và chỉ số Firebase
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

# Phân tích nhóm: chỉ cộng 4 chỉ số per-user/new-user; showrate liệt kê theo từng ad unit
def analyze_group(agg_df: pd.DataFrame, prefix: str, version_a: str, version_b: str) -> Optional[Dict]:
    if agg_df is None or agg_df.empty:
        return None

    item_col = "ad_name" if "ad_name" in agg_df.columns else "ad_unit"

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

    def sums_per_user(d: pd.DataFrame) -> Dict[str, float]:
        if d is None or d.empty:
            return dict(imp_user=np.nan, imp_new_user=np.nan, rev_user=np.nan, rev_new_user=np.nan)
        user = pd.to_numeric(d.get("user", np.nan), errors="coerce")
        new_user = pd.to_numeric(d.get("new_user", np.nan), errors="coerce")
        imp = pd.to_numeric(d.get("impressions", np.nan), errors="coerce")
        rev = pd.to_numeric(d.get("estimated_earnings", np.nan), errors="coerce")
        imp_user = imp.divide(user).where(user > 0)
        imp_new_user = imp.divide(new_user).where(new_user > 0)
        rev_user = rev.divide(user).where(user > 0)
        rev_new_user = rev.divide(new_user).where(new_user > 0)
        return dict(
            imp_user=np.nansum(imp_user),
            imp_new_user=np.nansum(imp_new_user),
            rev_user=np.nansum(rev_user),
            rev_new_user=np.nansum(rev_new_user),
        )

    def per_item_showrate_requests(d: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        if d is None or d.empty:
            return {}
        g = d.groupby(item_col, dropna=False)[["requests", "impressions"]].sum(min_count=1).reset_index()
        g["show_rate"] = g["impressions"].divide(g["requests"]).where(g["requests"] > 0)
        out = {}
        for _, r in g.iterrows():
            name = str(r[item_col])
            out[name] = dict(
                requests=float(r.get("requests", np.nan)),
                show_rate=float(r.get("show_rate", np.nan)),
            )
        return out

    data = {
        "A": {
            "sums": sums_per_user(A),
            "by_item": per_item_showrate_requests(A),
        },
        "B": {
            "sums": sums_per_user(B),
            "by_item": per_item_showrate_requests(B),
        },
        "item_col": item_col,
    }
    return data

def analysis_to_text(prefix: str, label: str, data: Dict, version_a: str, version_b: str):
    A = data["A"]; B = data["B"]
    sums_pairs = [
        ("imp/user", A["sums"].get("imp_user"), B["sums"].get("imp_user"), "num2"),
        ("imp/new user", A["sums"].get("imp_new_user"), B["sums"].get("imp_new_user"), "num2"),
        ("rev/user", A["sums"].get("rev_user"), B["sums"].get("rev_user"), "num6"),
        ("rev/new user", A["sums"].get("rev_new_user"), B["sums"].get("rev_new_user"), "num6"),
    ]
    def fmt(v, kind):
        if kind == "num2": return format_num2(v)
        if kind == "num6": return format_num6(v)
        if kind == "pct":  return format_pct(v)
        return str(v)

    st.markdown(f"• Nhóm: {label} (prefix: {prefix})")

    for title, va, vb, kind in sums_pairs:
        if pd.isna(va) and pd.isna(vb):
            continue
        if pd.notna(vb) and pd.notna(va):
            trend = "tốt hơn" if vb > va else ("kém hơn" if vb < va else "bằng")
        elif pd.notna(vb) and pd.isna(va):
            trend = "tốt hơn"
        elif pd.isna(vb) and pd.notna(va):
            trend = "kém hơn"
        else:
            trend = "bằng"
        st.write(f"- {title}: {version_a} = {fmt(va, kind)} → {version_b} = {fmt(vb, kind)} ({trend}).")

    names = sorted(set(B["by_item"].keys()) | set(A["by_item"].keys()))
    if names:
        st.write("- Chi tiết show rate theo từng ad unit:")
        for name in names:
            srA = A["by_item"].get(name, {}).get("show_rate", np.nan)
            rqA = A["by_item"].get(name, {}).get("requests", np.nan)
            srB = B["by_item"].get(name, {}).get("show_rate", np.nan)
            rqB = B["by_item"].get(name, {}).get("requests", np.nan)
            if pd.notna(srB) and pd.notna(srA):
                trend = "tốt hơn" if srB > srA else ("kém hơn" if srB < srA else "bằng")
            elif pd.notna(srB) and pd.isna(srA):
                trend = "tốt hơn"
            elif pd.isna(srB) and pd.notna(srA):
                trend = "kém hơn"
            else:
                trend = "bằng"
            st.write(
                f"  • {name} {version_a} showrate {format_pct(srA)} (req {int(rqA) if pd.notna(rqA) else '—'}) "
                f"{'<' if trend=='kém hơn' else ('>' if trend=='tốt hơn' else '=')} "
                f"{version_b} {format_pct(srB)} (req {int(rqB) if pd.notna(rqB) else '—'}) — {trend}."
            )

    st.divider()

# ------- Daily compare helpers (Checkver) với Decimal chuẩn cent -------
def _pair_key(ver_a: str, ver_b: str) -> str:
    return f"{str(ver_a)}__{str(ver_b)}"

def to_cent(val) -> Decimal:
    if val is None:
        return Decimal("0.00")
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return Decimal("0.00")
    if "." in s and "," in s:
        s = s.replace(",", "")
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        try:
            d = Decimal(str(float(val)))
        except Exception:
            d = Decimal("0.00")
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def _fmt_mdy(date_iso: str) -> str:
    dt = pd.to_datetime(date_iso, errors="coerce")
    if pd.isna(dt):
        return str(date_iso)
    return f"{dt.month}/{dt.day}/{dt.year}"

def _init_daily_dates(df_src: pd.DataFrame, version_a: str, version_b: str, n_default: int = 4) -> list:
    if df_src is None or df_src.empty or "date" not in df_src.columns or "version" not in df_src.columns:
        today = pd.Timestamp.today().normalize()
        base = [(today - pd.Timedelta(days=i)).date().isoformat() for i in range(n_default)][::-1]
        return base
    d = df_src.copy()
    d = d[d["version"].astype(str).isin([str(version_a), str(version_b)])]
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[d["date"].notna()]
    if d.empty:
        today = pd.Timestamp.today().normalize()
        base = [(today - pd.Timedelta(days=i)).date().isoformat() for i in range(n_default)][::-1]
        return base
    uniq = sorted(pd.to_datetime(d["date"].dt.normalize()).dt.date.unique())
    pick = uniq[-n_default:] if len(uniq) >= n_default else uniq
    return [x.isoformat() for x in pick]

def _daily_revenue_map(df_src: pd.DataFrame, version_list: List[str]) -> Dict[Tuple[str, str], float]:
    out_dec: Dict[Tuple[str, str], Decimal] = {}
    if df_src is None or df_src.empty:
        return {}
    need = {"version", "date", "estimated_earnings"}
    if not need.issubset(set(df_src.columns)):
        return {}
    d = df_src.copy()
    d["version"] = d["version"].astype(str)
    d = d[d["version"].isin([str(v) for v in version_list])]
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[d["date"].notna()]
    d["date_iso"] = d["date"].dt.date.astype(str)

    for _, r in d.iterrows():
        key = (str(r["version"]), str(r["date_iso"]))
        amount = to_cent(r.get("estimated_earnings", 0))
        out_dec[key] = out_dec.get(key, Decimal("0.00")) + amount

    return {k: float(v) for k, v in out_dec.items()}

def _fb_totals_for_version(fb_df: Optional[pd.DataFrame], df_src: pd.DataFrame, version: str) -> tuple:
    if not isinstance(fb_df, pd.DataFrame) or fb_df.empty or "version" not in fb_df.columns:
        return (None, None)
    v = str(version)
    fb = fb_df.copy()
    fb["version"] = fb["version"].astype(str)
    fb = fb[fb["version"] == v]
    if "app" in fb.columns and "app" in df_src.columns:
        apps = df_src[df_src["version"].astype(str) == v]["app"].dropna().astype(str).unique().tolist()
        if apps:
            fb = fb[fb["app"].astype(str).isin(apps)]
    if fb.empty:
        return (None, None)
    u = pd.to_numeric(fb.get("user", np.nan), errors="coerce").sum(min_count=1)
    nu = pd.to_numeric(fb.get("new_user", np.nan), errors="coerce").sum(min_count=1)
    u = float(u) if pd.notna(u) else None
    nu = float(nu) if pd.notna(nu) else None
    return (u, nu)

def render_daily_checkver_table(
    df_src: pd.DataFrame,
    fb_df: Optional[pd.DataFrame],
    version_a: str,
    version_b: str,
    section_title: str = "Bảng theo ngày (rev auto từ AdMob; nhập user/new user)"
):
    st.markdown("---")
    with st.expander(section_title, expanded=False):
        pair_key = _pair_key(version_a, version_b)
        st.session_state.setdefault("checkver_daycols", {})
        if pair_key not in st.session_state["checkver_daycols"]:
            st.session_state["checkver_daycols"][pair_key] = _init_daily_dates(df_src, version_a, version_b, n_default=4)

        day_cols = st.session_state["checkver_daycols"][pair_key]  # list date_iso
        n = len(day_cols)

        rev_map = _daily_revenue_map(df_src, [version_a, version_b])
        user_total_a, _ = _fb_totals_for_version(fb_df, df_src, version_a)
        user_total_b, _ = _fb_totals_for_version(fb_df, df_src, version_b)

        # Header
        cols = st.columns([1] + [1]*n + [0.2, 1, 0.8])
        cols[0].markdown(" ")
        for i, d in enumerate(day_cols):
            cols[1+i].markdown(f"**{_fmt_mdy(d)}**")
        cols[1+n].markdown(" ")
        cols[2+n].markdown("**Tổng**")
        with cols[3+n]:
            if st.button("➕ Thêm ngày", key=f"btn_add_day_{pair_key}"):
                if len(day_cols) > 0:
                    last = max(pd.to_datetime(x) for x in day_cols)
                    nxt = (last + timedelta(days=1)).date().isoformat()
                else:
                    nxt = pd.Timestamp.today().date().isoformat()
                day_cols.append(nxt)
                st.session_state["checkver_daycols"][pair_key] = day_cols
                safe_rerun()

        def block_for_version(version: str, fb_total_user: Optional[float], tint: str):
            st.markdown(f"**Phiên bản {version}**")
            # Row 1: rev
            row1 = st.columns([1] + [1]*n + [0.2, 1, 0.8])
            row1[0].markdown("rev")
            total_rev_dec = Decimal("0.00")
            rev_inputs = []
            for i, d in enumerate(day_cols):
                default_rev = float(rev_map.get((str(version), d), 0.0))
                val = row1[1+i].number_input(
                    label=f"rev_{version}_{d}",
                    value=float(to_cent(default_rev)),
                    step=0.01,
                    format="%.2f",
                    key=f"rev_in_{version}_{d}",
                )
                rev_inputs.append(val)
                total_rev_dec += to_cent(val)
            row1[1+n].markdown(" ")
            total_rev = float(total_rev_dec)
            row1[2+n].markdown(f"**{total_rev:,.2f}**")

            # Row 2: user
            row2 = st.columns([1] + [1]*n + [0.2, 1, 0.8])
            row2[0].markdown(f"<div style='background:{tint};padding:2px 6px;border-radius:4px'>user</div>", unsafe_allow_html=True)
            user_inputs = []
            for i, d in enumerate(day_cols):
                u = row2[1+i].number_input(
                    label=f"user_{version}_{d}",
                    value=0.0,
                    step=1.0,
                    format="%.0f",
                    key=f"user_in_{version}_{d}",
                )
                user_inputs.append(u)
            row2[1+n].markdown(" ")
            shown_user_total = fb_total_user if fb_total_user is not None else float(np.nansum(user_inputs))
            row2[2+n].markdown(f"**{shown_user_total:,.0f}**" if pd.notna(shown_user_total) else "**—**")

            # Row 3: rev/user
            row3 = st.columns([1] + [1]*n + [0.2, 1, 0.8])
            row3[0].markdown("rev/user (USD)")
            rpu_list = []
            for i in range(n):
                v = (rev_inputs[i] / user_inputs[i]) if (user_inputs[i] and user_inputs[i] > 0) else np.nan
                rpu_list.append(v)
                row3[1+i].markdown(f"{v:,.6f}" if pd.notna(v) else "—")
            row3[1+n].markdown(" ")
            rpu_total = (total_rev / shown_user_total) if (shown_user_total and shown_user_total > 0) else np.nan
            row3[2+n].markdown(f"**{rpu_total:,.6f}**" if pd.notna(rpu_total) else "**—**")

            return dict(
                rev_inputs=rev_inputs,
                user_inputs=user_inputs,
                rpu_list=rpu_list,
                rpu_total=rpu_total,
                total_rev=total_rev,
                shown_user_total=shown_user_total,
            )

        st.markdown(" ")
        data_a = block_for_version(version_a, user_total_a, tint="#F6EBD9")
        st.markdown(" ")
        data_b = block_for_version(version_b, user_total_b, tint="#FFF2CC")

        # Row cuối: % thay đổi rev/user (B so A) — xanh nếu dương, cam nếu âm
        st.markdown(" ")
        cols_change = st.columns([1] + [1]*n + [0.2, 1, 0.8])
        cols_change[0].markdown(f"**Thay đổi rev/user {version_b} so {version_a} (%)**")

        def pct_badge(pct_val: float) -> str:
            if pd.isna(pct_val):
                return "—"
            color = "#22c55e" if pct_val > 0 else ("#fb923c" if pct_val < 0 else "#e2e8f0")
            return f"<div style='background:{color};color:#111827;padding:2px 6px;border-radius:4px;text-align:center'>{(pct_val*100):.2f}%</div>"

        for i in range(n):
            a_rpu = data_a["rpu_list"][i]
            b_rpu = data_b["rpu_list"][i]
            pct = ((b_rpu - a_rpu) / a_rpu) if (pd.notna(a_rpu) and a_rpu > 0) else np.nan
            cols_change[1+i].markdown(pct_badge(pct), unsafe_allow_html=True)

        cols_change[1+n].markdown(" ")
        a_total_rpu = data_a["rpu_total"]
        b_total_rpu = data_b["rpu_total"]
        pct_total = ((b_total_rpu - a_total_rpu) / a_total_rpu) if (pd.notna(a_total_rpu) and a_total_rpu > 0) else np.nan
        cols_change[2+n].markdown(pct_badge(pct_total), unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Checkver — So sánh 2 version (1 tệp)")

    with st.expander("Hướng dẫn", expanded=True):
        st.markdown(
            """
- B1: Upload CSV Admob (dùng chung với tab Manual).
- B2: Upload Firebase (Version, Active/New users) rồi bấm Nạp Firebase.
- B3: Bật mapping nếu muốn hiển thị ad name.
- B4: Chọn Version A (cũ) và Version B (mới). Có thể bật “Checkver template” để chỉ highlight các nhóm phân tích.
- B5: Bấm “Phân tích checkver” để xem phân tích chi tiết từng nhóm.
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
            template_mode = st.checkbox("Checkver template (chỉ highlight các nhóm phân tích)", value=False)

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
            df_view["imp_per_user"]     = df_view["impressions"].astype(float).divide(user).where(user > 0)
            df_view["imp_per_new_user"] = df_view["impressions"].astype(float).divide(new_user).where(new_user > 0)
            df_view["rev_per_user"]     = df_view["estimated_earnings"].astype(float).divide(user).where(user > 0)
            df_view["rev_per_new_user"] = df_view["estimated_earnings"].astype(float).divide(new_user).where(new_user > 0)
            df_view["req_per_user"]     = df_view["requests"].astype(float).divide(user).where(user > 0)
            df_view["req_per_new_user"] = df_view["requests"].astype(float).divide(new_user).where(new_user > 0)

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

            # Xác định id_cols và allowed_keys (template)
            if template_mode:
                prefixes = tuple(p[0] for p in ANALYZE_GROUPS)
                mask_pref = df_view["ad_unit"].astype(str).str.lower().str.startswith(prefixes)
                if "ad_name" in df_view.columns:
                    mask_pref = mask_pref | df_view["ad_name"].astype(str).str.lower().str.startswith(prefixes)
                id_cols = [c for c in ["app", "ad_unit"] if c in df_view.columns]
                if not id_cols:
                    id_cols = [c for c in ["app", "ad_name"] if c in df_view.columns]
                if not id_cols:
                    id_cols = ["ad_unit"] if "ad_unit" in df_view.columns else ["ad_name"]
                allowed_keys = set(tuple(row[c] if c in df_view.columns else None for c in id_cols)
                                   for _, row in df_view[mask_pref].iterrows())
            else:
                id_cols = [c for c in ["app", "ad_unit"] if c in df_view.columns]
                if not id_cols:
                    id_cols = [c for c in ["app", "ad_name"] if c in df_view.columns]
                if not id_cols:
                    id_cols = ["ad_unit"] if "ad_unit" in df_view.columns else ["ad_name"]
                allowed_keys = None

            print_cols = list(df_print.columns)
            cell_colors = build_checkver_cell_colors(
                df_numeric=df_view.reset_index(drop=True),
                print_cols=print_cols,
                version_a=version_a,
                version_b=version_b,
                id_cols=id_cols,
                shade_b_rows=bool(shade_b),
                allowed_keys=allowed_keys,
            )
            render_table(df_print, table_height, cell_colors=cell_colors)

            # Export
            xls = to_excel_bytes({"Checkver": df_view[ordered_cols]})
            st.download_button(
                "Tải Excel (theo bảng hiện tại)",
                data=xls,
                file_name=f"checkver_{version_a}_vs_{version_b}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # ===== BẢNG THEO NGÀY: rev auto từ AdMob; nhập user/new user; so sánh % theo rev/user =====
            df_src_for_daily = st.session_state.get("global_df")
            fb_df_for_daily = st.session_state.get("firebase_df")
            render_daily_checkver_table(
                df_src=df_src_for_daily,
                fb_df=fb_df_for_daily,
                version_a=version_a,
                version_b=version_b,
                section_title="Bảng theo ngày (rev auto từ AdMob; nhập user/new user)"
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

                any_res = False
                for prefix, label in ANALYZE_GROUPS:
                    data = analyze_group(base, prefix, version_a, version_b)
                    if data is None:
                        continue
                    any_res = True
                    analysis_to_text(prefix, label, data, version_a, version_b)

                if not any_res:
                    st.info("Version B không có ads thuộc các nhóm trong danh sách phân tích.")