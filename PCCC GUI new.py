import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="SWCNT ATPE PCCC", layout="wide")
st.title("SWCNT ATPE PCCC Explorer")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["-", "—", "–", ""])
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"chirality", "doc_pct", "sc_pct", "pccc", "n_h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    for c in ["doc_pct", "sc_pct", "pccc", "n_h", "pccc_sigma", "n_h_sigma"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["chirality", "doc_pct", "sc_pct", "pccc", "n_h"])
    df = df[(df["pccc"] > 0) & (df["n_h"] > 0)]
    return df

def hill_curve(x: np.ndarray, pccc: float, n_h: float) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return 1.0 / (1.0 + np.power(pccc / x, n_h))

def nearest_point(sub: pd.DataFrame, doc_target: float, sc_target: float) -> pd.Series:
    # simple Euclidean distance in (doc, sc) space
    d = np.sqrt((sub["doc_pct"] - doc_target)**2 + (sub["sc_pct"] - sc_target)**2)
    idx = d.idxmin()
    row = sub.loc[idx].copy()
    row["_dist"] = float(d.loc[idx])
    return row

# ----- Sidebar -----
st.sidebar.header("Controls")
csv_path = st.sidebar.text_input("CSV file path", value="pccc_parameters.csv")

df = load_data(csv_path)

chirality_options = sorted(df["chirality"].unique().tolist())
selected_ch = st.sidebar.multiselect(
    "Select chirality (n,m)",
    options=chirality_options,
    default=chirality_options[:3] if len(chirality_options) >= 3 else chirality_options
)

# slider ranges from data (still allows any value within range)
doc_min, doc_max = float(df["doc_pct"].min()), float(df["doc_pct"].max())
sc_min, sc_max = float(df["sc_pct"].min()), float(df["sc_pct"].max())

doc_target = st.sidebar.slider("DOC (%)", min_value=doc_min, max_value=doc_max, value=min(0.10, doc_max), step=0.01)
sc_target  = st.sidebar.slider("SC (%)",  min_value=sc_min,  max_value=sc_max,  value=min(0.90, sc_max),  step=0.01)

show_50 = st.sidebar.checkbox("Show 50% level line", value=True)

# optional: snap mode only for now (keeps behavior honest)
snap_mode = st.sidebar.selectbox("Condition matching", ["Snap to nearest experimental point"], index=0)

# ----- Plot -----
x = np.linspace(0.01, 3.0, 800)
fig = go.Figure()

if show_50:
    fig.add_trace(go.Scatter(
        x=x, y=np.full_like(x, 0.5),
        mode="lines",
        name="50% level",
        line=dict(dash="dash")
    ))

rows_used = []
for ch in selected_ch:
    sub = df[df["chirality"] == ch]
    if sub.empty:
        continue

    row = nearest_point(sub, doc_target, sc_target)
    pccc = float(row["pccc"])
    n_h  = float(row["n_h"])
    y = hill_curve(x, pccc, n_h)

    label = f"{ch}  DOC={row['doc_pct']:.2f}%  SC={row['sc_pct']:.2f}%  (PCCC={pccc:g}, n_H={n_h:g})"
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label))

    rows_used.append({
        "chirality": ch,
        "doc_target": doc_target,
        "sc_target": sc_target,
        "doc_used": float(row["doc_pct"]),
        "sc_used": float(row["sc_pct"]),
        "distance": float(row["_dist"]),
        "pccc": pccc,
        "n_h": n_h,
        "pccc_sigma": row.get("pccc_sigma", np.nan),
        "n_h_sigma": row.get("n_h_sigma", np.nan),
    })

fig.update_layout(
    xaxis_title="SDS (%)",
    yaxis_title="Normalized Concentration",
    xaxis=dict(range=[0.01, 3.0]),
    yaxis=dict(range=[-0.05, 1.05]),
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Parameters used (nearest experimental points)")
if rows_used:
    used_df = pd.DataFrame(rows_used).sort_values("chirality").reset_index(drop=True)
    st.dataframe(used_df, use_container_width=True)

    # clear warning when you're far from any measured point
    if (used_df["distance"] > 0.05).any():
        st.warning("Some selections are far from existing experimental points. Currently the app snaps to nearest data; consider adding more DOC/SC data or implementing interpolation.")
else:
    st.info("No chirality selected or no matching data found.")

st.caption("Hill model: y = 1 / (1 + (PCCC/x)^(n_H)), x is SDS (%).")
