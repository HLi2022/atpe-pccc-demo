import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Voigt needs SciPy ---
try:
    from scipy.special import wofz
except Exception:
    st.error("SciPy is required for Voigt profiles. Run: conda run -n atpe_app pip install scipy")
    st.stop()

st.set_page_config(page_title="ATPE App", layout="wide")
st.title("ATPE Application")

PCCC_PATH_DEFAULT = "data/pccc_parameters.csv"
SEMI_PATH_DEFAULT = "data/swcnt_info_semiconducting.csv"
METALLIC_PATH_DEFAULT = "data/swcnt_info_metallic.csv"

@st.cache_data
def load_pccc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["-", "—", "–", ""])
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"chirality", "doc_pct", "sc_pct", "pccc", "n_h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pccc_parameters.csv missing columns: {sorted(missing)}")

    for c in ["doc_pct", "sc_pct", "pccc", "pccc_sigma", "n_h", "n_h_sigma"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["chirality", "doc_pct", "sc_pct", "pccc", "n_h"])
    
    # Extract n, m
    # Expecting format "(n,m)"
    try:
        nm = df["chirality"].str.extract(r"\((\d+),(\d+)\)")
        df["n"] = pd.to_numeric(nm[0])
        df["m"] = pd.to_numeric(nm[1])
    except Exception:
        pass # If format doesn't match, n/m columns might be NaN which is fine for basic plotting but limits advanced selection
        
    df = df[(df["pccc"] > 0) & (df["n_h"] > 0)]
    return df

@st.cache_data
def load_semiconducting(path: str) -> pd.DataFrame:
    # Semiconducting: No header, so we provide names matching the file structure
    # Based on verify_csv: columns 0..10
    # Expected: 0:diameter, 1:chiral_angle, 2:mod_type, 3: ?(M11-?), 4: ?(M11+?), 5:RBM, 6:wl11(S11), 7:wl22(S22), 8:n, 9:m, 10:SDS%
    # But wait, looking at user request/file content:
    # "3.36923,29.3249,1,0.330377,0.595727,78,3752.83,2081.24,25,24,0.0722224DS%"
    # 25,24 are n,m ?
    # Let's assume standard order often seen or inferred:
    # d, theta, mod, ?, ?, rbm, E11, E22, n, m, ...
    
    # Let's map explicitly based on the head output we saw earlier:
    # 3.36923 (d), 29.3249 (theta), 1 (mod), ... , 3752.83 (E11), 2081.24 (E22), 25 (n), 24 (m)
    # The columns seem to be:
    # 0: diameter
    # 1: chiral_angle
    # 2: mod_type
    # 3: ?
    # 4: ?
    # 5: RBM
    # 6: wl11 (S11)
    # 7: wl22 (S22)
    # 8: index_n
    # 9: index_m
    # 10: SDS%
    
    df = pd.read_csv(path, header=None,  na_values=["-", "—", "–", ""])
    # Assign likely column names
    col_names = [
        "diameter", "chiral_angle", "mod_type", "col3", "col4", 
        "rbm", "wl11", "wl22", "index_n", "index_m", "sds_pct"
    ]
    # Handle if file has more or fewer columns slightly (sanity check)
    if len(df.columns) == len(col_names):
        df.columns = col_names
    else:
        # Fallback or error - but let's try to map the critical ones by index if length differs
        # Assuming fixed structure for now based on user info
        # Only map what we need if size matches roughly
        if len(df.columns) >= 10:
             df = df.iloc[:, :11]
             df.columns = col_names[:len(df.columns)]
    
    # Process
    # Ensure n, m are numeric
    df["index_n"] = pd.to_numeric(df["index_n"], errors="coerce")
    df["index_m"] = pd.to_numeric(df["index_m"], errors="coerce")
    
    # Ensure wavelengths and other params are numeric
    for c in ["diameter", "chiral_angle", "rbm", "wl11", "wl22", "sds_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    df = df.dropna(subset=["index_n", "index_m"])
    
    df["chirality"] = df.apply(lambda r: f"({int(r['index_n'])},{int(r['index_m'])})", axis=1)
    df["n"] = df["index_n"].astype(int)
    df["m"] = df["index_m"].astype(int)
    df["type"] = "Semiconducting"
    
    # Sanity check (B)
    # 1. wl11 < wl22 check (general rule for S11/S22 in range we care about?)
    #    Actually S11 (E11) is longer wavelength than S22 (E22). 
    #    E.g. (6,5) E11~975nm, E22~570nm. So wl11 > wl22 usually. 
    #    Wait, load_semiconducting mapped columns: wl11, wl22.
    #    Let's check consistent range: 300 < wl < 4000.
    
    warnings = []
    if not df.empty:
        # Check range
        wls = pd.concat([df["wl11"], df["wl22"]])
        if (wls < 300).any() or (wls > 4000).any():
             warnings.append("Some semiconducting wavelengths are outside 300-4000nm range.")
             
        # Check E11 > E22 usually? 
        # let's just warn if ratio is weird? 
        # Actually simplest is just range check for "sanity".
    
    if warnings:
        for w in warnings:
            st.warning(f"Semiconducting Data Warning: {w}")

    return df

@st.cache_data
def load_metallic(path: str) -> pd.DataFrame:
    # Metallic: Has header
    # diameter,chiral_angle,mod_type,M11-,M11+,RBM,wl11-,wl11+,index_n,index_m,SDS%
    df = pd.read_csv(path, na_values=["-", "—", "–", ""])
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Map to standard keys
    # wl11- -> M11- (user wants display as M11-)
    # wl11+ -> M11+
    
    # Robustness against header repetition or bad rows
    if "index_n" in df.columns:
         df["index_n"] = pd.to_numeric(df["index_n"], errors="coerce")
    if "index_m" in df.columns:
         df["index_m"] = pd.to_numeric(df["index_m"], errors="coerce")

    # Rename to clean internal names (C)
    rename_map = {"wl11-": "m11_minus", "wl11+": "m11_plus", 
                  "wl11": "m11_single"} # maybe wl11 is used for armchair in some formats
    df = df.rename(columns=rename_map)
    
    # Ensure new columns are numeric if they exist
    for c in ["m11_minus", "m11_plus", "m11_single"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    df["chirality"] = df.apply(lambda r: f"({int(r['index_n'])},{int(r['index_m'])})", axis=1)
    df["n"] = df["index_n"].astype(int)
    df["m"] = df["index_m"].astype(int)
    df["type"] = "Metallic"
    return df

def hill_curve(x: np.ndarray, pccc: float, n_h: float) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return 1.0 / (1.0 + np.power(pccc / x, n_h))

def nearest_row(sub: pd.DataFrame, doc_target: float, sc_target: float) -> pd.Series:
    d = np.sqrt((sub["doc_pct"] - doc_target) ** 2 + (sub["sc_pct"] - sc_target) ** 2)
    idx = d.idxmin()
    row = sub.loc[idx].copy()
    row["_dist"] = float(d.loc[idx])
    return row

def voigt_profile(x: np.ndarray, mu: float, sigma: float, gamma: float) -> np.ndarray:
    # normalized Voigt profile
    z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

def simulate_uvvis(x_nm: np.ndarray, wl_e11: float, wl_e22: float,
                   sigma: float, gamma: float,
                   amp_e11: float, amp_e22: float) -> np.ndarray:
    y = amp_e11 * voigt_profile(x_nm, wl_e11, sigma, gamma)
    if not np.isnan(wl_e22) and wl_e22 > 0:
        y += amp_e22 * voigt_profile(x_nm, wl_e22, sigma, gamma)
    return y

# ---- Sidebar ----
st.sidebar.header("Inputs")

pccc_path = st.sidebar.text_input("PCCC CSV", value=PCCC_PATH_DEFAULT)
semi_path = st.sidebar.text_input("Semiconducting CSV", value=SEMI_PATH_DEFAULT)
met_path = st.sidebar.text_input("Metallic CSV", value=METALLIC_PATH_DEFAULT)

if not os.path.exists(pccc_path):
    st.error(f"Cannot find: {pccc_path}")
    st.stop()
if not os.path.exists(semi_path):
    st.error(f"Cannot find: {semi_path}")
    st.stop()
if not os.path.exists(met_path):
    st.error(f"Cannot find: {met_path}")
    st.stop()

try:
    pccc_df = load_pccc(pccc_path)
    # (A) Fixed double call
    semi_df = load_semiconducting(semi_path)
    met_df = load_metallic(met_path)
    # Combine for chirality lookups
    all_info_df = pd.concat([semi_df, met_df], ignore_index=True)
except Exception as e:
    st.error(str(e))
    st.stop()

# Union of all available chiralities for selection
all_chiralities = sorted(list(set(pccc_df["chirality"].tolist()) | set(all_info_df["chirality"].tolist())))
chirality_options = all_chiralities

# -- Interactive Selection Logic --
if "selected_list" not in st.session_state:
    # Initialize with some default if available
    st.session_state.selected_list = chirality_options[:3] if len(chirality_options) >= 3 else chirality_options

st.sidebar.subheader("Select Chirality")

# 1. Add by n, m
c1, c2, c3 = st.sidebar.columns([1, 1, 1])
with c1:
    # Use info_df as the primary source for (n,m) structure since it likely contains more
    # But we want to ensure we cover any unique ones in pccc_df too (less likely but possible)
    # Let's build a small unique dataframe of n,m from info_df for the dropdowns
    # (Assuming info_df covers the physical existence of tubes better)
    
    nm_source = all_info_df[["n", "m"]].drop_duplicates().sort_values(["n", "m"])
    
    if not nm_source.empty:
        available_n = sorted(nm_source["n"].unique())
        sel_n = st.selectbox("n", available_n)
        
        # Filter m based on n
        available_m = sorted(nm_source[nm_source["n"] == sel_n]["m"].unique())
        sel_m = st.selectbox("m", available_m)
    else:
        sel_n, sel_m = None, None

with c3:
    # Button to add
    st.write("") # Spacer
    st.write("") 
    if st.button("Add"):
        to_add = f"({sel_n},{sel_m})"
        if to_add in chirality_options:
             if to_add not in st.session_state.selected_list:
                st.session_state.selected_list.append(to_add)
                # Force update the widget state to reflect the addition
                st.session_state.multiselect_widget = st.session_state.selected_list
        else:
             st.warning(f"Chirality {to_add} not found in database.")

# 2. Multiselect to show/remove
def update_selection():
    st.session_state.selected_list = st.session_state.multiselect_widget

# Ensure state is synced before rendering
if "multiselect_widget" not in st.session_state:
    st.session_state.multiselect_widget = st.session_state.selected_list

selected_ch = st.sidebar.multiselect(
    "Selected Chiralities",
    options=chirality_options,
    key="multiselect_widget",
    on_change=update_selection
)

doc_min, doc_max = float(pccc_df["doc_pct"].min()), float(pccc_df["doc_pct"].max())
sc_min, sc_max = float(pccc_df["sc_pct"].min()), float(pccc_df["sc_pct"].max())

if doc_min < doc_max:
    doc_target = st.sidebar.slider("DOC (%)", min_value=doc_min, max_value=doc_max, value=float(doc_min), step=0.01)
else:
    st.sidebar.markdown(f"**DOC (%):** {doc_min}")
    doc_target = doc_min

if sc_min < sc_max:
    sc_target = st.sidebar.slider("SC (%)",  min_value=sc_min,  max_value=sc_max,  value=float(sc_max), step=0.01)
else:
    st.sidebar.markdown(f"**SC (%):** {sc_min}")
    sc_target = sc_min

show_50 = st.sidebar.checkbox("Show 50% level line", value=True)

st.sidebar.subheader("UVVis (Voigt) settings")
sigma_nm = st.sidebar.slider("sigma (nm)", min_value=1.0, max_value=30.0, value=8.0, step=0.5)
gamma_nm = st.sidebar.slider("gamma (nm)", min_value=0.5, max_value=30.0, value=6.0, step=0.5)
amp_e11 = st.sidebar.slider("E11 amplitude", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
amp_e22 = st.sidebar.slider("E22 amplitude", min_value=0.1, max_value=3.0, value=0.6, step=0.1)
amp_met = st.sidebar.slider("Metallic Amplitude", min_value=0.1, max_value=3.0, value=0.6, step=0.1)

# ---- Build plots ----
x_sds = np.linspace(0.01, 3.0, 800)

# UVVis x-range based on selected tubes
def get_uv_range(selected):
    sub = all_info_df[all_info_df["chirality"].isin(selected)]
    if sub.empty:
        return 400.0, 1400.0
    
    # Collect all relevant wavelengths
    vals = []
    # Semi: wl11, wl22
    semi_sub = sub[sub["type"] == "Semiconducting"]
    if not semi_sub.empty:
        vals.extend(semi_sub["wl11"].dropna().values)
        vals.extend(semi_sub["wl22"].dropna().values)
        
    # Met: m11_minus, m11_plus (mapped from check in load_metallic)
    met_sub = sub[sub["type"] == "Metallic"]
    if not met_sub.empty:
        # Check col names
        if "m11_minus" in met_sub.columns: vals.extend(met_sub["m11_minus"].dropna().values)
        if "m11_plus" in met_sub.columns: vals.extend(met_sub["m11_plus"].dropna().values)
        if "m11_single" in met_sub.columns: vals.extend(met_sub["m11_single"].dropna().values)

    wls = np.array(vals)
    # (E) Robustness for empty selection or missing data
    if len(wls) == 0:
         return 400.0, 1400.0
         
    wmin = float(np.nanmin(wls)) - 80
    wmax = float(np.nanmax(wls)) + 80
    wmin = max(250.0, wmin)
    wmax = min(2000.0, wmax)
    return wmin, wmax

wl_min, wl_max = get_uv_range(selected_ch)
x_nm = np.linspace(wl_min, wl_max, 2000)

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=False,
    vertical_spacing=0.12,
    subplot_titles=("PCCC (Hill) vs SDS", "Simulated UV-Vis absorption (Voigt: E11 + E22)")
)

# PCCC: 50% line
if show_50:
    fig.add_trace(
        go.Scatter(x=x_sds, y=np.full_like(x_sds, 0.5), mode="lines",
                   name="50% level", line=dict(dash="dash")),
        row=1, col=1
    )

rows_used_semi = []
rows_used_met = []

for ch in selected_ch:
    # 1. PCCC
    sub_p = pccc_df[pccc_df["chirality"] == ch]
    
    pccc_val = np.nan
    n_h_val = np.nan
    doc_used_val = np.nan
    sc_used_val = np.nan
    dist_val = np.nan
    pccc_sigma_val = np.nan
    n_h_sigma_val = np.nan

    if not sub_p.empty:
        r = nearest_row(sub_p, doc_target, sc_target)
        pccc_val = float(r["pccc"])
        n_h_val = float(r["n_h"])
        y = hill_curve(x_sds, pccc_val, n_h_val)

        fig.add_trace(
            go.Scatter(
                x=x_sds, y=y, mode="lines",
                name=f"{ch} (PCCC={pccc_val:g}, n_H={n_h_val:g})",
                legend="legend"
            ),
            row=1, col=1
        )
        
        doc_used_val = float(r["doc_pct"])
        sc_used_val = float(r["sc_pct"])
        dist_val = float(r["_dist"])
        pccc_sigma_val = r.get("pccc_sigma", np.nan)
        n_h_sigma_val = r.get("n_h_sigma", np.nan)

    # 2. UVVis
    # 2. UVVis
    # Check if Semi or Metallic
    is_semi = False
    is_met = False
    
    sub_semi = semi_df[semi_df["chirality"] == ch]
    sub_met = met_df[met_df["chirality"] == ch]
    
    # Setup for table row
    row_data = {
            "chirality": ch,
            "doc_target": doc_target,
            "sc_target": sc_target,
            "doc_used": doc_used_val,
            "sc_used": sc_used_val,
            "distance": dist_val,
            "pccc": pccc_val,
            "pccc_sigma": pccc_sigma_val,
            "n_h": n_h_val,
            "n_h_sigma": n_h_sigma_val,
    }

    if not sub_semi.empty:
        is_semi = True
        row = sub_semi.iloc[0]
        wl11 = float(row["wl11"])
        wl22 = float(row["wl22"])
        
        # (D) Abstract UVVis plotting with peaks list
        peaks = []
        if wl11 > 0: peaks.append({"label": "S11", "wl": wl11, "amp": amp_e11})
        if wl22 > 0: peaks.append({"label": "S22", "wl": wl22, "amp": amp_e22})

        # Add to Semi Table
        row_data["S11"] = wl11
        row_data["S22"] = wl22
        row_data["diameter_nm"] = row.get("diameter", np.nan)
        row_data["chiral_angle_deg"] = row.get("chiral_angle", np.nan)
        rows_used_semi.append(row_data)

    elif not sub_met.empty:
        is_met = True
        row = sub_met.iloc[0]
        
        # Metallic logic with new columns
        # Armchair might rely on m11_minus only or m11_single
        w1 = float(row.get("m11_minus", np.nan))
        w2 = float(row.get("m11_plus", np.nan))
        
        peaks = []
        if not np.isnan(w1) and w1 > 0: peaks.append({"label": "M11-", "wl": w1, "amp": amp_met})
        if not np.isnan(w2) and w2 > 0: peaks.append({"label": "M11+", "wl": w2, "amp": amp_met})
        
        # Add to Met Table
        row_data["M11-"] = w1
        row_data["M11+"] = w2
        row_data["diameter_nm"] = row.get("diameter", np.nan)
        row_data["chiral_angle_deg"] = row.get("chiral_angle", np.nan)
        rows_used_met.append(row_data)

    # Plotting loop abstracted
    if is_semi or is_met:
        # Sum curve
        y_total = np.zeros_like(x_nm)
        for p in peaks:
            y_total += p["amp"] * voigt_profile(x_nm, p["wl"], sigma_nm, gamma_nm)
            
        name_suffix = "(Semi)" if is_semi else "(Met)"
        fig.add_trace(go.Scatter(x=x_nm, y=y_total, mode="lines", name=f"{ch} {name_suffix}", legend="legend2"), row=2, col=1)
        
        # Markers
        for p in peaks:
             val_at_peak = 0
             # Re-calc sum at this exact peak for the marker y-value? 
             # Or just single peak contribution? Usually visualization shows total absorbance at that point.
             # Let's simple approximate or calc full sum:
             for sub_p in peaks:
                 val_at_peak += sub_p["amp"] * voigt_profile(np.array([p["wl"]]), sub_p["wl"], sigma_nm, gamma_nm)
                 
             fig.add_trace(go.Scatter(
                 x=[p["wl"]], y=val_at_peak,
                 mode="markers", 
                 name=f"{ch} {p['label']}", 
                 showlegend=False, 
                 legend="legend2",
                 hovertemplate=f"{ch} {p['label']}<br>Wl: %{{x:.1f}} nm"
             ), row=2, col=1)
        
    else:
        # Just PCCC data maybe?
        if not np.isnan(pccc_val):
             # Add to a generic list or maybe just skip detailed table?
             # Let's add to Semi table as fallback or skip
             pass


fig.update_xaxes(title_text="SDS (%)", range=[0.01, 3.0], row=1, col=1)
fig.update_yaxes(title_text="Normalized concentration", range=[-0.05, 1.05], row=1, col=1)

fig.update_xaxes(title_text="Wavelength (nm)", range=[wl_min, wl_max], row=2, col=1)
fig.update_yaxes(title_text="Absorbance (a.u., simulated)", row=2, col=1)

fig.update_layout(
    height=850,
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
    legend2=dict(x=1.02, y=0.45, xanchor="left", yanchor="top"),
)

st.plotly_chart(fig, use_container_width=True)

# ---- Info tables ----
st.subheader("Semiconducting Tubes")
if rows_used_semi:
    df_semi_res = pd.DataFrame(rows_used_semi)
    cols_semi = ["chirality", "diameter_nm", "chiral_angle_deg", "S11", "S22", 
                 "doc_used", "sc_used", "pccc", "pccc_sigma", "n_h", "n_h_sigma", "distance"]
    # Filter to existing
    cols_semi = [c for c in cols_semi if c in df_semi_res.columns]
    st.dataframe(df_semi_res[cols_semi].sort_values("chirality").reset_index(drop=True), use_container_width=True)
else:
    st.info("No Semiconducting tubes selected.")

st.subheader("Metallic Tubes")
if rows_used_met:
    df_met_res = pd.DataFrame(rows_used_met)
    cols_met = ["chirality", "diameter_nm", "chiral_angle_deg", "M11-", "M11+", 
                "doc_used", "sc_used", "pccc", "pccc_sigma", "n_h", "n_h_sigma", "distance"]
    cols_met = [c for c in cols_met if c in df_met_res.columns]
    st.dataframe(df_met_res[cols_met].sort_values("chirality").reset_index(drop=True), use_container_width=True)
else:
    st.info("No Metallic tubes selected.")
    
all_rows = rows_used_semi + rows_used_met
if all_rows:
    merged_dist = pd.DataFrame(all_rows)
    if not merged_dist.empty and "distance" in merged_dist.columns and (merged_dist["distance"] > 0.05).any():
        st.warning("Some selections are far from existing DOC/SC points. The app snaps to the nearest experimental point.")
else:
    st.info("Select at least one chirality.")

st.divider()
st.subheader("Raw data (optional)")
with st.expander("Show raw PCCC table"):
    st.dataframe(pccc_df, use_container_width=True)
with st.expander("Show raw Semiconducting table"):
    st.dataframe(semi_df, use_container_width=True)
with st.expander("Show raw Metallic table"):
    st.dataframe(met_df, use_container_width=True)
