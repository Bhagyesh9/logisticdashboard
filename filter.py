import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

st.set_page_config(page_title="HUL Logistics Filter", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ──
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        border: 2px solid #4472C4;
        padding: 14px 12px;
        border-radius: 10px;
        background: rgba(68,114,196,0.08);
        overflow: visible; min-width: 0;
    }
    div[data-testid="stMetric"] label { font-size: 0.8rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: clamp(0.9rem, 1.1vw, 1.25rem) !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    .section-header {
        background: #4472C4; color: white; padding: 10px 18px;
        border-radius: 8px; font-size: 1.1rem; font-weight: 600;
        margin: 20px 0 12px 0;
    }
    .insight-card {
        border: 1px solid #4472C4; border-radius: 8px;
        padding: 14px 18px; margin-bottom: 10px;
        background: rgba(68,114,196,0.05);
    }
    /* Hide deploy button, source code viewer, main menu, toolbar */
    .stAppDeployButton, #MainMenu,
    button[title="View app source"], .viewerBadge_container__r5tak,
    ._profileContainer_gzau3_53, [data-testid="stToolbar"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
    }
    /* Always keep sidebar fully expanded — remove ability to collapse */
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        transform: translateX(0) !important;
        width: 21rem !important;
        min-width: 21rem !important;
        max-width: 21rem !important;
        margin-left: 0 !important;
    }
    /* Hide any button that would collapse the sidebar */
    [data-testid="stSidebar"] button[kind="header"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebar"] button[aria-label*="close" i],
    [data-testid="stSidebar"] button[aria-label*="collapse" i] {
        display: none !important;
    }
    /* Ensure main content doesn't overlap sidebar */
    [data-testid="stAppViewContainer"] > .main {
        margin-left: 21rem !important;
    }
</style>
""", unsafe_allow_html=True)

PRODUCT_FILE = "product_names.parquet"

SHEET_PARQUETS = {
    "Leg Product Detail": "Leg_Product_Detail.parquet",
    "Leg Detail": "Leg_Detail.parquet",
    "Outlier Legs": "Outlier_Legs.parquet",
    "Route Summary": "Route_Summary.parquet",
    "Truck Type Analysis": "Truck_Type_Analysis.parquet",
    "Transporter Analysis": "Transporter_Analysis.parquet",
    "Cluster Information and Dispatc": "Cluster_Information_and_Dispatc.parquet",
    "Route Truck Wastage All": "Route_Truck_Wastage_All.parquet",
}


# ── Helpers ──

def fmt_rs(x):
    if pd.isna(x) or x == 0:
        return "-"
    if abs(x) >= 1e7:
        return f"Rs {x / 1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"Rs {x / 1e5:,.2f} L"
    return f"Rs {x:,.0f}"


def fmt_num(x):
    if pd.isna(x):
        return "-"
    return f"{x:,.0f}"


def fmt_wt(x):
    if pd.isna(x) or x == 0:
        return "-"
    return f"{x:,.1f} T"


def fmt_pct(x):
    if pd.isna(x):
        return "-"
    return f"{x:.1f}%"


def weighted_util(lpd, ld, group_col=None):
    """Compute weight-share weighted average utilisation.

    For each leg, material weight share = Net Weight / Leg Weight.
    Weighted Util = sum(Util% * material_weight) / sum(material_weight).
    If group_col is given, returns a Series indexed by group_col.
    Otherwise returns a single float.
    """
    if lpd.empty or ld.empty or "Utilisation %" not in ld.columns:
        return pd.Series(dtype=float) if group_col else 0.0

    # Get leg-level util and leg weight from Leg Detail
    leg_util = ld[["Leg ID", "Utilisation %", "Leg Weight (T)"]].drop_duplicates("Leg ID")

    # Merge with LPD to get material-level weight per leg
    m = lpd[["Leg ID", "Net Weight (T)"]].copy()
    if group_col and group_col in lpd.columns:
        m[group_col] = lpd[group_col]

    m = m.merge(leg_util, on="Leg ID", how="inner")
    m = m[m["Leg Weight (T)"] > 0]

    # Weight contribution = material's net weight in this leg
    # Weighted util = Util% * material_weight, then sum / sum(weights)
    m["weighted_util"] = m["Utilisation %"] * m["Net Weight (T)"]

    if group_col:
        grouped = m.groupby(group_col)
        return (grouped["weighted_util"].sum() / grouped["Net Weight (T)"].sum())
    else:
        total_wt = m["Net Weight (T)"].sum()
        if total_wt == 0:
            return 0.0
        return m["weighted_util"].sum() / total_wt


def load_product_names():
    return pd.read_parquet(PRODUCT_FILE)


@st.cache_data(show_spinner="Loading data...")
def load_analysis_data():
    sheets = {}
    for name, fname in SHEET_PARQUETS.items():
        if os.path.exists(fname):
            sheets[name] = pd.read_parquet(fname)
    return sheets


def compute_filtered_data(sheets, material_set, dispatch_filter, cluster_filter,
                          route_filter, truck_filter, month_filter=None, tdp_filter=None,
                          source_filter=None, dest_filter=None):
    """Filter all sheets by material, dispatch, cluster, route, truck, month, TDP, source, dest."""
    lpd = sheets.get("Leg Product Detail", pd.DataFrame())
    ld_raw = sheets.get("Leg Detail", pd.DataFrame())

    # Month/TDP live on Leg Detail — restrict to allowed leg IDs first
    leg_restrict = None
    if not ld_raw.empty:
        ld_mask = pd.Series(True, index=ld_raw.index)
        if month_filter and "Month" in ld_raw.columns:
            ld_mask &= ld_raw["Month"].astype(str).isin(month_filter)
        if tdp_filter and "TDP" in ld_raw.columns:
            ld_mask &= ld_raw["TDP"].astype(str).isin(tdp_filter)
        if month_filter or tdp_filter:
            leg_restrict = set(ld_raw.loc[ld_mask, "Leg ID"].unique()) if "Leg ID" in ld_raw.columns else set()

    if material_set and not lpd.empty and "Material" in lpd.columns:
        lpd_filtered = lpd[lpd["Material"].isin(material_set)]
    else:
        lpd_filtered = lpd

    if dispatch_filter and "Dispatch Type" in lpd_filtered.columns:
        lpd_filtered = lpd_filtered[lpd_filtered["Dispatch Type"].isin(dispatch_filter)]

    if cluster_filter and "Cluster Desc" in lpd_filtered.columns:
        lpd_filtered = lpd_filtered[lpd_filtered["Cluster Desc"].isin(cluster_filter)]

    if route_filter and "Route Code" in lpd_filtered.columns:
        lpd_filtered = lpd_filtered[lpd_filtered["Route Code"].isin(route_filter)]

    if truck_filter and "Truck Type" in lpd_filtered.columns:
        lpd_filtered = lpd_filtered[lpd_filtered["Truck Type"].isin(truck_filter)]

    if source_filter and "Sending Plant" in lpd_filtered.columns:
        lpd_filtered = lpd_filtered[lpd_filtered["Sending Plant"].isin(source_filter)]

    if dest_filter and "Receiving Plant" in lpd_filtered.columns:
        lpd_filtered = lpd_filtered[lpd_filtered["Receiving Plant"].isin(dest_filter)]

    if leg_restrict is not None and "Leg ID" in lpd_filtered.columns:
        lpd_filtered = lpd_filtered[lpd_filtered["Leg ID"].isin(leg_restrict)]

    # Extract join keys
    shipment_set = set(lpd_filtered["Shipment No"].unique()) if "Shipment No" in lpd_filtered.columns else set()
    route_code_set = set(lpd_filtered["Route Code"].unique()) if "Route Code" in lpd_filtered.columns else set()
    route_name_set = set(lpd_filtered["Route"].unique()) if "Route" in lpd_filtered.columns else set()
    leg_id_set = set(lpd_filtered["Leg ID"].unique()) if "Leg ID" in lpd_filtered.columns else set()
    cluster_code_set = set(lpd_filtered["Cluster Code"].dropna().unique()) if "Cluster Code" in lpd_filtered.columns else set()

    any_active = material_set or dispatch_filter or cluster_filter or route_filter or truck_filter or month_filter or tdp_filter or source_filter or dest_filter
    filtered = {"Leg Product Detail": lpd_filtered}

    for name in ["Leg Detail", "Outlier Legs"]:
        if name not in sheets:
            continue
        df = sheets[name]
        if any_active:
            mask = pd.Series(False, index=df.index)
            if "Shipment No" in df.columns and shipment_set:
                mask = mask | df["Shipment No"].isin(shipment_set)
            if "Leg ID" in df.columns and leg_id_set:
                mask = mask | df["Leg ID"].isin(leg_id_set)
            filtered[name] = df[mask]
        else:
            filtered[name] = df

    for name in ["Truck Type Analysis", "Transporter Analysis", "Route Truck Wastage All"]:
        if name not in sheets:
            continue
        df = sheets[name]
        if "Route Code" in df.columns and route_code_set and any_active:
            filtered[name] = df[df["Route Code"].isin(route_code_set)]
        else:
            filtered[name] = df

    if "Route Summary" in sheets:
        df = sheets["Route Summary"]
        if "Route" in df.columns and route_name_set and any_active:
            filtered["Route Summary"] = df[df["Route"].isin(route_name_set)]
        else:
            filtered["Route Summary"] = df

    cname = "Cluster Information and Dispatc"
    if cname in sheets:
        df = sheets[cname]
        if "Cluster Code" in df.columns and cluster_code_set and any_active:
            filtered[cname] = df[df["Cluster Code"].isin(cluster_code_set)]
        else:
            filtered[cname] = df

    return filtered


# ── Charts ──

CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Calibri, sans-serif", size=12),
    margin=dict(l=50, r=80, t=50, b=50),
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
COLORS = ["#1B2A4A", "#4472C4", "#5B9BD5", "#A5C8E1", "#ED7D31", "#FFC000",
          "#70AD47", "#9DC3E6", "#2F5597", "#BF8F00", "#548235", "#C55A11"]


def chart_waste_breakdown(ld):
    if ld.empty or "Waste Category" not in ld.columns or "Waste (Rs)" not in ld.columns:
        return None
    data = ld[ld["Waste (Rs)"] > 0].groupby("Waste Category", dropna=True)["Waste (Rs)"].sum().reset_index()
    if data.empty:
        return None
    fig = px.pie(data, values="Waste (Rs)", names="Waste Category",
                 color_discrete_sequence=COLORS, hole=0.45)
    fig.update_layout(**CHART_LAYOUT, title="Waste Breakdown by Category", showlegend=True)
    fig.update_traces(textinfo="percent+label", textfont_size=12)
    return fig


def chart_dd_vs_ndd_comparison(lpd, ld):
    """Dual-axis bar: DD vs NDD — Spend on left axis, Waste on right axis."""
    if lpd.empty or "Dispatch Type" not in lpd.columns:
        return None
    spend = lpd.groupby("Dispatch Type").agg(
        Spend=("Leg Cost (Rs)", "sum"),
    ).reset_index()
    spend = spend[spend["Dispatch Type"].isin(["DD", "NDD"])]
    if spend.empty:
        return None

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Total Spend (Rs)", "Total Waste (Rs)"),
                        horizontal_spacing=0.15)

    fig.add_trace(go.Bar(x=spend["Dispatch Type"], y=spend["Spend"],
                         marker_color=["#1B2A4A", "#4472C4"],
                         text=spend["Spend"].apply(lambda x: f"{x/1e7:,.1f} Cr"),
                         textposition="outside", showlegend=False), row=1, col=1)

    if not ld.empty and "Waste (Rs)" in ld.columns and "Dispatch Type" in ld.columns:
        waste = ld.groupby("Dispatch Type")["Waste (Rs)"].sum().reset_index()
        waste = waste[waste["Dispatch Type"].isin(["DD", "NDD"])]
        if not waste.empty:
            fig.add_trace(go.Bar(x=waste["Dispatch Type"], y=waste["Waste (Rs)"],
                                 marker_color=["#ED7D31", "#FFC000"],
                                 text=waste["Waste (Rs)"].apply(lambda x: f"{x/1e7:,.1f} Cr"),
                                 textposition="outside", showlegend=False), row=1, col=2)

    fig.update_layout(**CHART_LAYOUT, title_text="DD vs NDD Comparison")
    fig.update_yaxes(tickformat=",", row=1, col=1)
    fig.update_yaxes(tickformat=",", row=1, col=2)
    return fig


def chart_dd_vs_ndd_metrics(lpd, ld):
    """Build a DD vs NDD metrics comparison dataframe."""
    if lpd.empty or "Dispatch Type" not in lpd.columns:
        return None
    # Weighted util by dispatch type
    dt_util = weighted_util(lpd, ld, group_col="Dispatch Type")
    rows = []
    for dt in ["DD", "NDD"]:
        lp = lpd[lpd["Dispatch Type"] == dt] if "Dispatch Type" in lpd.columns else pd.DataFrame()
        lg = ld[ld["Dispatch Type"] == dt] if not ld.empty and "Dispatch Type" in ld.columns else pd.DataFrame()
        shipments = len(lg)
        skus = int(lg["SKU Count"].sum()) if "SKU Count" in lg.columns and not lg.empty else 0
        weight = lp["Net Weight (T)"].sum() if "Net Weight (T)" in lp.columns else 0
        spend = lp["Leg Cost (Rs)"].sum() if "Leg Cost (Rs)" in lp.columns else 0
        waste = lg["Waste (Rs)"].sum() if "Waste (Rs)" in lg.columns and not lg.empty else 0
        util = dt_util.get(dt, 0) if isinstance(dt_util, pd.Series) else 0
        cpt = spend / weight if weight > 0 else 0
        rows.append({"Dispatch": dt, "Shipments": shipments, "SKUs": skus,
                      "Weight (T)": weight, "Spend (Rs)": spend, "Waste (Rs)": waste,
                      "Waste %": (waste / spend * 100) if spend > 0 else 0,
                      "Avg Util %": util, "Avg CPT": cpt})
    df = pd.DataFrame(rows)
    total = {"Dispatch": "Total", "Shipments": df["Shipments"].sum(), "SKUs": df["SKUs"].sum(),
             "Weight (T)": df["Weight (T)"].sum(),
             "Spend (Rs)": df["Spend (Rs)"].sum(), "Waste (Rs)": df["Waste (Rs)"].sum()}
    total["Waste %"] = (total["Waste (Rs)"] / total["Spend (Rs)"] * 100) if total["Spend (Rs)"] > 0 else 0
    total["Avg Util %"] = df["Avg Util %"].mean()
    total["Avg CPT"] = total["Spend (Rs)"] / total["Weight (T)"] if total["Weight (T)"] > 0 else 0
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    return df


def chart_cluster_insights(lpd, ld):
    """Build cluster-wise insights table."""
    if lpd.empty or "Cluster Desc" not in lpd.columns:
        return None
    cl_lpd = lpd.groupby("Cluster Desc", dropna=True).agg(
        Weight=("Net Weight (T)", "sum"), Spend=("Leg Cost (Rs)", "sum"),
    ).reset_index()

    if not ld.empty and "Cluster Desc" in ld.columns:
        cl_ld = ld.groupby("Cluster Desc", dropna=True).agg(**{
            "Shipments": ("Leg ID", "count"),
            "SKUs": ("SKU Count", "sum") if "SKU Count" in ld.columns else ("Leg ID", "count"),
            "Waste": ("Waste (Rs)", "sum") if "Waste (Rs)" in ld.columns else ("Leg ID", "count"),
        }).reset_index()
        # Weighted utilisation by cluster
        cl_util = weighted_util(lpd, ld, group_col="Cluster Desc")
        cl_util = cl_util.reset_index()
        cl_util.columns = ["Cluster Desc", "Util"]
        cl = cl_lpd.merge(cl_ld, on="Cluster Desc", how="left").fillna(0)
        cl = cl.merge(cl_util, on="Cluster Desc", how="left").fillna(0)
    else:
        cl = cl_lpd
        cl["Shipments"] = 0
        cl["SKUs"] = 0
        cl["Waste"] = 0
        cl["Util"] = 0

    cl["Waste %"] = cl["Waste"] / cl["Spend"].replace(0, np.nan) * 100
    cl["CPT"] = cl["Spend"] / cl["Weight"].replace(0, np.nan)
    cl = cl.sort_values("Spend", ascending=False)
    cl = cl[["Cluster Desc", "Shipments", "SKUs", "Weight", "Spend", "Waste", "Waste %", "Util", "CPT"]]
    cl.columns = ["Cluster (State)", "Shipments", "SKUs", "Weight (T)", "Spend (Rs)",
                   "Waste (Rs)", "Waste %", "Avg Util %", "Avg CPT"]
    return cl


def chart_cluster_bar(cl_df):
    """Dual bar: spend on top, waste % on bottom by cluster."""
    if cl_df is None or cl_df.empty:
        return None
    data = cl_df.head(12).copy()
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Spend by Cluster", "Waste % by Cluster"),
                        vertical_spacing=0.22, row_heights=[0.55, 0.45])
    # Spend bars
    fig.add_trace(go.Bar(x=data["Cluster (State)"], y=data["Spend (Rs)"],
                         marker_color="#1B2A4A", showlegend=False,
                         text=data["Spend (Rs)"].apply(lambda x: f"{x/1e7:,.1f}Cr"),
                         textposition="outside", textfont_size=9), row=1, col=1)
    # Waste % bars
    fig.add_trace(go.Bar(x=data["Cluster (State)"], y=data["Waste %"],
                         marker_color="#ED7D31", showlegend=False,
                         text=data["Waste %"].apply(lambda x: f"{x:.1f}%"),
                         textposition="outside", textfont_size=9), row=2, col=1)
    layout = {k: v for k, v in CHART_LAYOUT.items() if k != "height"}
    fig.update_layout(**layout, height=520)
    fig.update_xaxes(tickangle=-40, tickfont_size=9)
    fig.update_yaxes(tickformat=",", row=1, col=1)
    return fig


def chart_top_waste_routes(ld, top_n=10):
    if ld.empty or "Route" not in ld.columns or "Waste (Rs)" not in ld.columns:
        return None
    data = ld.groupby("Route", dropna=True)["Waste (Rs)"].sum().nlargest(top_n).reset_index()
    if data.empty:
        return None
    data["Route_Short"] = data["Route"].str[:40]
    fig = px.bar(data, x="Waste (Rs)", y="Route_Short", orientation="h",
                 color_discrete_sequence=["#4472C4"],
                 text=data["Waste (Rs)"].apply(lambda x: f"{x/1e5:,.1f}L"))
    layout = {k: v for k, v in CHART_LAYOUT.items() if k != "height"}
    fig.update_layout(**layout, title=f"Top {top_n} Routes by Waste",
                      yaxis=dict(autorange="reversed", tickfont_size=9),
                      height=450, margin_r=60, margin_l=200,
                      yaxis_title="", xaxis_tickformat=",")
    fig.update_traces(textposition="outside", textfont_size=10)
    return fig


def chart_route_truck_heatmap(ld):
    """Heatmap: waste by route x truck type (top 15 routes x top 10 trucks)."""
    if ld.empty or "Route" not in ld.columns or "Truck Type" not in ld.columns or "Waste (Rs)" not in ld.columns:
        return None
    pivot = ld.pivot_table(values="Waste (Rs)", index="Route", columns="Truck Type",
                           aggfunc="sum", fill_value=0)
    if pivot.empty:
        return None
    # Top routes by total waste
    top_routes = pivot.sum(axis=1).nlargest(15).index
    # Top truck types by total waste
    top_trucks = pivot.sum(axis=0).nlargest(10).index
    sub = pivot.loc[pivot.index.isin(top_routes), pivot.columns.isin(top_trucks)]
    sub = sub.loc[sub.sum(axis=1).sort_values(ascending=False).index]
    # Truncate route names
    sub.index = [r[:40] for r in sub.index]
    fig = px.imshow(sub, color_continuous_scale=["#f0f4ff", "#4472C4", "#1B2A4A"],
                    labels=dict(x="Truck Type", y="Route", color="Waste (Rs)"),
                    aspect="auto")
    layout = {k: v for k, v in CHART_LAYOUT.items() if k != "height"}
    fig.update_layout(**layout, title="Route x Truck Type Waste Heatmap",
                      height=520, margin_l=200,
                      yaxis=dict(tickfont_size=9),
                      xaxis=dict(tickfont_size=10, side="bottom"))
    return fig


def chart_utilisation_dist(ld):
    if ld.empty or "Utilisation %" not in ld.columns:
        return None
    util = ld["Utilisation %"].dropna()
    if util.empty:
        return None
    fig = px.histogram(util, nbins=25, color_discrete_sequence=["#2C3E6B"],
                       labels={"value": "Utilisation %", "count": "Shipments"})
    fig.update_layout(**CHART_LAYOUT, title="Truck Utilisation Distribution",
                      xaxis_title="Utilisation %", xaxis_ticksuffix="%",
                      yaxis_title="Shipments", showlegend=False)
    return fig


def chart_month_end_vs_mid(lpd, ld):
    """Bar: Month-End vs Rest of Month comparison."""
    if ld.empty or "Is this month end?" not in ld.columns:
        return None
    rows = []
    # Tag LPD with month-end flag via Leg ID
    if "Is this month end?" in ld.columns:
        me_map = ld[["Leg ID", "Is this month end?"]].drop_duplicates("Leg ID")
        lpd_me = lpd.merge(me_map, on="Leg ID", how="left") if "Leg ID" in lpd.columns else pd.DataFrame()
    else:
        lpd_me = pd.DataFrame()

    for period, label in [("Month End", "Month End"), ("Mid Month", "Mid Month")]:
        sub = ld[ld["Is this month end?"] == period] if period == "Month End" else ld[ld["Is this month end?"] != "Month End"]
        sub_lpd = lpd_me[lpd_me["Is this month end?"] == period] if period == "Month End" and not lpd_me.empty else (
            lpd_me[lpd_me["Is this month end?"] != "Month End"] if not lpd_me.empty else pd.DataFrame())
        shipments = len(sub)
        skus = int(sub["SKU Count"].sum()) if "SKU Count" in sub.columns else 0
        spend = sub["Leg Cost (Rs)"].sum() if "Leg Cost (Rs)" in sub.columns else 0
        waste = sub["Waste (Rs)"].sum() if "Waste (Rs)" in sub.columns else 0
        util = weighted_util(sub_lpd, sub) if not sub_lpd.empty else 0
        rows.append({"Period": label, "Shipments": shipments, "SKUs": skus,
                      "Spend (Rs)": spend, "Waste (Rs)": waste,
                      "Waste %": (waste/spend*100) if spend > 0 else 0,
                      "Avg Util %": util})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Spend", x=df["Period"], y=df["Spend (Rs)"],
                         marker_color="#1B2A4A", text=df["Spend (Rs)"].apply(lambda x: f"{x:,.0f}"),
                         textposition="outside"))
    fig.add_trace(go.Bar(name="Waste", x=df["Period"], y=df["Waste (Rs)"],
                         marker_color="#ED7D31", text=df["Waste (Rs)"].apply(lambda x: f"{x:,.0f}"),
                         textposition="outside"))
    fig.update_layout(**CHART_LAYOUT, title="Month-End vs Rest: Spend & Waste",
                      barmode="group", yaxis_tickformat=",",
                      legend=dict(orientation="h", y=-0.15))
    return fig, df


def chart_tdp_performance(lpd, ld):
    """Line: waste % by TDP."""
    if ld.empty or "TDP" not in ld.columns or "Waste (Rs)" not in ld.columns:
        return None
    grp = ld.groupby("TDP", dropna=True).agg(
        Spend=("Leg Cost (Rs)", "sum"), Waste=("Waste (Rs)", "sum"),
    ).reset_index()
    # Weighted util by TDP
    if "TDP" in ld.columns:
        tdp_map = ld[["Leg ID", "TDP"]].drop_duplicates("Leg ID")
        lpd_tdp = lpd.merge(tdp_map, on="Leg ID", how="inner") if "Leg ID" in lpd.columns else pd.DataFrame()
        tdp_util = weighted_util(lpd_tdp, ld, group_col="TDP") if not lpd_tdp.empty else pd.Series(dtype=float)
        tdp_util = tdp_util.reset_index()
        tdp_util.columns = ["TDP", "Util"]
        grp = grp.merge(tdp_util, on="TDP", how="left").fillna(0)
    else:
        grp["Util"] = 0
    grp["Waste %"] = grp["Waste"] / grp["Spend"].replace(0, np.nan) * 100
    grp = grp.sort_values("TDP")
    if grp.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grp["TDP"], y=grp["Waste %"], name="Waste %",
                             mode="lines+markers", line=dict(color="#ED7D31", width=2.5),
                             marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=grp["TDP"], y=grp["Util"], name="Util %",
                             mode="lines+markers", line=dict(color="#4472C4", width=2.5),
                             marker=dict(size=7), yaxis="y2"))
    fig.update_layout(**CHART_LAYOUT, title="TDP Performance: Waste % & Utilisation",
                      yaxis=dict(title="Waste %", ticksuffix="%"),
                      yaxis2=dict(title="Util %", overlaying="y", side="right", ticksuffix="%"),
                      xaxis=dict(tickangle=-45, tickfont_size=9, dtick=2),
                      legend=dict(orientation="h", y=-0.22))
    return fig


def chart_monthly_trend(ld):
    if ld.empty or "Month" not in ld.columns:
        return None
    tmp = ld.copy()
    tmp["Month"] = tmp["Month"].astype(str)
    grp = tmp.groupby("Month", dropna=True).agg(
        Legs=("Leg ID", "count"),
        Waste=("Waste (Rs)", "sum") if "Waste (Rs)" in tmp.columns else ("Leg ID", "count"),
    ).reset_index().sort_values("Month")
    if grp.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grp["Month"], y=grp["Legs"], name="Legs",
                             mode="lines+markers", line=dict(color="#1B2A4A", width=2.5), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=grp["Month"], y=grp["Waste"], name="Waste (Rs)",
                             mode="lines+markers", line=dict(color="#ED7D31", width=2.5),
                             marker=dict(size=7), yaxis="y2"))
    fig.update_layout(**CHART_LAYOUT, title="Monthly Trend: Legs & Waste",
                      yaxis=dict(title="Legs", tickformat=","),
                      yaxis2=dict(title="Waste (Rs)", overlaying="y", side="right", tickformat=","),
                      legend=dict(orientation="h", y=-0.18))
    return fig


def chart_brand_spend(brand_summary):
    if brand_summary.empty:
        return None
    data = brand_summary.nlargest(12, "Total Spend (Rs)")
    data["Brand_Short"] = data["Brand"].str[:25]
    fig = px.bar(data, x="Total Spend (Rs)", y="Brand_Short", orientation="h",
                 color_discrete_sequence=["#2F5597"],
                 text=data["Total Spend (Rs)"].apply(lambda x: f"{x/1e7:,.1f}Cr" if x >= 1e7 else f"{x/1e5:,.1f}L"))
    layout = {k: v for k, v in CHART_LAYOUT.items() if k != "height"}
    fig.update_layout(**layout, title="Spend by Brand (Top 12)", height=450,
                      yaxis=dict(autorange="reversed", tickfont_size=9),
                      yaxis_title="", xaxis_tickformat=",", margin_l=160)
    fig.update_traces(textposition="outside", textfont_size=10)
    return fig


def chart_brand_waste(brand_summary):
    if brand_summary.empty:
        return None
    data = brand_summary.nlargest(12, "Total Waste (Rs)")
    data["Brand_Short"] = data["Brand"].str[:25]
    fig = px.bar(data, x="Total Waste (Rs)", y="Brand_Short", orientation="h",
                 color_discrete_sequence=["#ED7D31"],
                 text=data["Total Waste (Rs)"].apply(lambda x: f"{x/1e7:,.1f}Cr" if x >= 1e7 else f"{x/1e5:,.1f}L"))
    layout = {k: v for k, v in CHART_LAYOUT.items() if k != "height"}
    fig.update_layout(**layout, title="Waste by Brand (Top 12)", height=450,
                      yaxis=dict(autorange="reversed", tickfont_size=9),
                      yaxis_title="", xaxis_tickformat=",", margin_l=160)
    fig.update_traces(textposition="outside", textfont_size=10)
    return fig


def top_transporters_summary(ld, top_n=5):
    """Top-N transporters by spend from filtered Leg Detail."""
    if ld.empty or "Transporter" not in ld.columns:
        return None, []
    grp = ld.groupby("Transporter", dropna=True).agg(
        Legs=("Leg ID", "count"),
        Weight=("Leg Weight (T)", "sum"),
        Spend=("Leg Cost (Rs)", "sum"),
        Waste=("Waste (Rs)", "sum") if "Waste (Rs)" in ld.columns else ("Leg ID", "count"),
        Util=("Utilisation %", "mean") if "Utilisation %" in ld.columns else ("Leg ID", "count"),
    ).reset_index()
    grp["Waste %"] = grp["Waste"] / grp["Spend"].replace(0, np.nan) * 100
    grp["CPT"] = grp["Spend"] / grp["Weight"].replace(0, np.nan)
    grp = grp.sort_values("Spend", ascending=False).head(top_n)
    grp.columns = ["Transporter", "Legs", "Weight (T)", "Spend (Rs)", "Waste (Rs)",
                   "Avg Util %", "Waste %", "Avg CPT"]
    grp = grp[["Transporter", "Legs", "Weight (T)", "Spend (Rs)", "Waste (Rs)",
               "Waste %", "Avg Util %", "Avg CPT"]]
    return grp, grp["Transporter"].tolist()


def chart_transporter_truck_mix(ld, top_transporters):
    """Stacked bar: spend per transporter broken down by truck type."""
    if ld.empty or not top_transporters or "Truck Type" not in ld.columns:
        return None
    sub = ld[ld["Transporter"].isin(top_transporters)]
    if sub.empty:
        return None
    pivot = sub.groupby(["Transporter", "Truck Type"], dropna=True)["Leg Cost (Rs)"].sum().reset_index()
    fig = px.bar(pivot, x="Transporter", y="Leg Cost (Rs)", color="Truck Type",
                 color_discrete_sequence=COLORS, barmode="stack",
                 category_orders={"Transporter": top_transporters})
    fig.update_layout(**CHART_LAYOUT, title="Top 5 Transporters — Spend by Truck Type",
                      yaxis_tickformat=",", xaxis_tickangle=-20,
                      legend=dict(orientation="h", y=-0.25))
    return fig


def transporter_truck_breakdown(ld, top_transporters):
    """Per-transporter truck-wise performance table."""
    if ld.empty or not top_transporters or "Truck Type" not in ld.columns:
        return None
    sub = ld[ld["Transporter"].isin(top_transporters)]
    if sub.empty:
        return None
    grp = sub.groupby(["Transporter", "Truck Type"], dropna=True).agg(
        Legs=("Leg ID", "count"),
        Weight=("Leg Weight (T)", "sum"),
        Spend=("Leg Cost (Rs)", "sum"),
        Waste=("Waste (Rs)", "sum") if "Waste (Rs)" in sub.columns else ("Leg ID", "count"),
        Util=("Utilisation %", "mean") if "Utilisation %" in sub.columns else ("Leg ID", "count"),
    ).reset_index()
    grp["Waste %"] = grp["Waste"] / grp["Spend"].replace(0, np.nan) * 100
    grp["CPT"] = grp["Spend"] / grp["Weight"].replace(0, np.nan)
    grp.columns = ["Transporter", "Truck Type", "Legs", "Weight (T)", "Spend (Rs)",
                   "Waste (Rs)", "Avg Util %", "Waste %", "Avg CPT"]
    grp = grp[["Transporter", "Truck Type", "Legs", "Weight (T)", "Spend (Rs)",
               "Waste (Rs)", "Waste %", "Avg Util %", "Avg CPT"]]
    # Sort by transporter order (by rank), then spend desc within transporter
    grp["_rank"] = grp["Transporter"].map({t: i for i, t in enumerate(top_transporters)})
    grp = grp.sort_values(["_rank", "Spend (Rs)"], ascending=[True, False]).drop(columns="_rank")
    return grp


def show_sheet_tab(name, df):
    st.caption(f"{len(df):,} rows")
    display_limit = 5_000
    show_df = df.head(display_limit) if len(df) > display_limit else df
    if len(df) > display_limit:
        st.info(f"Showing first {display_limit:,} of {len(df):,} rows. Download CSV for full data.")
    st.dataframe(show_df, use_container_width=True, hide_index=True)
    st.download_button(
        f"Download {name} CSV",
        df.to_csv(index=False).encode("utf-8"),
        f"{name.replace(' ', '_')}_filtered.csv",
        "text/csv",
        key=f"dl_{name}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

st.title("HUL Logistics — Global Filter Dashboard")

products = load_product_names()
sheets = load_analysis_data()

# ── Sidebar: Logo + Product Filters (cascading) ──
if os.path.exists("hul_logo.png"):
    st.sidebar.image("hul_logo.png", use_container_width=True)
    st.sidebar.divider()
st.sidebar.header("Product Filters")
pf = products.copy()

FILTER_CASCADE = [
    ("Category", "Category"),
    ("Major Brand", "Major Brand"),
    ("Brand", "Brand (Mapping)"),
    ("Brand Code Desc", "Brand Code Des"),
    ("Material", "Material"),
]

selections = {}
for label, col in FILTER_CASCADE:
    if col not in pf.columns:
        selections[col] = []
        continue
    options = sorted(pf[col].dropna().unique())
    chosen = st.sidebar.multiselect(label, options, default=[], placeholder=f"All {label.lower()}s")
    selections[col] = chosen
    if chosen:
        pf = pf[pf[col].isin(chosen)]

any_product_filter = any(v for v in selections.values())
if any_product_filter:
    material_set = set(pf["Material"].dropna().unique()) if "Material" in pf.columns else set()
else:
    material_set = None

# ── Sidebar: Logistics Filters (cross-filtering cascade) ──
st.sidebar.header("Logistics Filters")

lpd_raw = sheets.get("Leg Product Detail", pd.DataFrame())
ld_raw = sheets.get("Leg Detail", pd.DataFrame())


def _plant_label_map(df, code_col, desc_col):
    if df.empty or code_col not in df.columns or desc_col not in df.columns:
        return {}
    pairs = df[[code_col, desc_col]].dropna().drop_duplicates(code_col)
    return dict(zip(pairs[code_col].astype(str), pairs[desc_col].astype(str)))


src_desc_map = _plant_label_map(ld_raw, "Sending Plant", "Sending Desc")
dst_desc_map = _plant_label_map(ld_raw, "Receiving Plant", "Receiving Desc")

# Cascade space: LD restricted by product filter via leg IDs
ld_space = ld_raw.copy()
if material_set is not None and "Leg ID" in lpd_raw.columns and "Material" in lpd_raw.columns:
    mat_legs = set(lpd_raw.loc[lpd_raw["Material"].isin(material_set), "Leg ID"].unique())
    ld_space = ld_space[ld_space["Leg ID"].isin(mat_legs)]

# Read previous selections (by widget key); source/dest store "CODE — DESC" labels
_src_prev = [l.split(" — ")[0] for l in st.session_state.get("f_source", [])]
_dst_prev = [l.split(" — ")[0] for l in st.session_state.get("f_dest", [])]
_SELECTIONS = {
    "Dispatch Type":   st.session_state.get("f_dispatch", []),
    "Cluster Desc":    st.session_state.get("f_cluster", []),
    "Route Code":      st.session_state.get("f_route", []),
    "Truck Type":      st.session_state.get("f_truck", []),
    "Sending Plant":   _src_prev,
    "Receiving Plant": _dst_prev,
    "Month":           st.session_state.get("f_month", []),
    "TDP":             st.session_state.get("f_tdp", []),
}


def _opts(col):
    df = ld_space
    for c, vals in _SELECTIONS.items():
        if c == col or not vals or c not in df.columns:
            continue
        df = df[df[c].astype(str).isin([str(v) for v in vals])]
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().astype(str).unique())


def _sanitize(key, valid):
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in valid]


# Dispatch Type
disp_avail = set(_opts("Dispatch Type"))
disp_opts = [d for d in ["DD", "NDD"] if d in disp_avail]
_sanitize("f_dispatch", disp_opts)
dispatch_filter = st.sidebar.multiselect(
    "Dispatch Type", disp_opts, key="f_dispatch", placeholder="All (DD + NDD)"
)

# Cluster
cluster_opts = _opts("Cluster Desc")
_sanitize("f_cluster", cluster_opts)
cluster_filter = st.sidebar.multiselect(
    "Cluster (State)", cluster_opts, key="f_cluster", placeholder="All clusters"
)

# Route
route_opts = _opts("Route Code")
_sanitize("f_route", route_opts)
route_filter = st.sidebar.multiselect(
    "Route", route_opts, key="f_route", placeholder="All routes"
)

# Truck Type
truck_opts = _opts("Truck Type")
_sanitize("f_truck", truck_opts)
truck_filter = st.sidebar.multiselect(
    "Truck Type", truck_opts, key="f_truck", placeholder="All truck types"
)

# Source (labels: "CODE — DESC")
src_codes_avail = _opts("Sending Plant")
src_labels_avail = [f"{c} — {src_desc_map[c]}" if c in src_desc_map else c for c in src_codes_avail]
_sanitize("f_source", src_labels_avail)
source_sel_labels = st.sidebar.multiselect(
    "Source (Sending Plant)", src_labels_avail, key="f_source", placeholder="All sources"
)
source_filter = [lbl.split(" — ")[0] for lbl in source_sel_labels]

# Destination
dst_codes_avail = _opts("Receiving Plant")
dst_labels_avail = [f"{c} — {dst_desc_map[c]}" if c in dst_desc_map else c for c in dst_codes_avail]
_sanitize("f_dest", dst_labels_avail)
dest_sel_labels = st.sidebar.multiselect(
    "Destination (Receiving Plant)", dst_labels_avail, key="f_dest", placeholder="All destinations"
)
dest_filter = [lbl.split(" — ")[0] for lbl in dest_sel_labels]

# Month
month_opts = _opts("Month")
_sanitize("f_month", month_opts)
month_filter = st.sidebar.multiselect(
    "Month", month_opts, key="f_month", placeholder="All months"
)

# TDP
tdp_opts = _opts("TDP")
_sanitize("f_tdp", tdp_opts)
tdp_filter = st.sidebar.multiselect(
    "TDP", tdp_opts, key="f_tdp", placeholder="All TDPs"
)

# Sidebar summary
st.sidebar.divider()
active_filters = []
if material_set is not None:
    active_filters.append(f"**{len(material_set):,}** material(s)")
if dispatch_filter:
    active_filters.append(f"Dispatch: **{', '.join(dispatch_filter)}**")
if cluster_filter:
    active_filters.append(f"Cluster: **{len(cluster_filter)}** selected")
if route_filter:
    active_filters.append(f"Route: **{len(route_filter)}** selected")
if truck_filter:
    active_filters.append(f"Truck: **{', '.join(truck_filter)}**")
if source_filter:
    active_filters.append(f"Source: **{len(source_filter)}** selected")
if dest_filter:
    active_filters.append(f"Dest: **{len(dest_filter)}** selected")
if month_filter:
    active_filters.append(f"Month: **{len(month_filter)}** selected")
if tdp_filter:
    active_filters.append(f"TDP: **{len(tdp_filter)}** selected")
if active_filters:
    st.sidebar.info(" | ".join(active_filters))
else:
    st.sidebar.info("No filter applied - showing all data")

if st.sidebar.button("Reset Filters"):
    for k in ["f_dispatch", "f_cluster", "f_route", "f_truck",
              "f_source", "f_dest", "f_month", "f_tdp"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# ── Apply all filters ──
filtered = compute_filtered_data(sheets, material_set, dispatch_filter, cluster_filter,
                                 route_filter, truck_filter, month_filter, tdp_filter,
                                 source_filter, dest_filter)

lpd = filtered.get("Leg Product Detail", pd.DataFrame())
ld = filtered.get("Leg Detail", pd.DataFrame())

# ── KPI Metrics ──
# Shipments = rows in Leg Detail (each row = one truck movement A→B)
# SKUs Moved = sum of SKU Count in Leg Detail
lpd_weight = lpd["Net Weight (T)"].sum() if "Net Weight (T)" in lpd.columns else 0
lpd_spend = lpd["Leg Cost (Rs)"].sum() if "Leg Cost (Rs)" in lpd.columns else 0
total_shipments = len(ld) if not ld.empty else 0
total_skus = int(ld["SKU Count"].sum()) if "SKU Count" in ld.columns and not ld.empty else 0
total_waste = ld["Waste (Rs)"].sum() if "Waste (Rs)" in ld.columns and not ld.empty else 0

kpi_row1 = st.columns(3)
kpi_row1[0].metric("Shipments", fmt_num(total_shipments))
kpi_row1[1].metric("SKUs Moved", fmt_num(total_skus))
kpi_row1[2].metric("Materials", fmt_num(lpd["Material"].nunique()) if "Material" in lpd.columns else "0")

kpi_row2 = st.columns(3)
kpi_row2[0].metric("Total Weight (T)", fmt_wt(lpd_weight))
kpi_row2[1].metric("Total Spend (Rs)", fmt_rs(lpd_spend))
kpi_row2[2].metric("Total Waste (Rs)", fmt_rs(total_waste))

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 1. DD vs NDD COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">1. DD vs NDD Comparison</div>', unsafe_allow_html=True)

ddndd_c1, ddndd_c2 = st.columns([1, 1])
with ddndd_c1:
    fig = chart_dd_vs_ndd_comparison(lpd, ld)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
with ddndd_c2:
    dd_table = chart_dd_vs_ndd_metrics(lpd, ld)
    if dd_table is not None:
        disp = dd_table.copy()
        disp["Shipments"] = disp["Shipments"].apply(fmt_num)
        disp["SKUs"] = disp["SKUs"].apply(fmt_num)
        disp["Weight (T)"] = disp["Weight (T)"].apply(lambda x: f"{x:,.1f}")
        disp["Spend (Rs)"] = disp["Spend (Rs)"].apply(fmt_rs)
        disp["Waste (Rs)"] = disp["Waste (Rs)"].apply(fmt_rs)
        disp["Waste %"] = disp["Waste %"].apply(lambda x: f"{x:.2f}%")
        disp["Avg Util %"] = disp["Avg Util %"].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")
        disp["Avg CPT"] = disp["Avg CPT"].apply(lambda x: f"{x:,.1f}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 2. CLUSTER (STATE) INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">2. Cluster (State) Insights</div>', unsafe_allow_html=True)

cl_df = chart_cluster_insights(lpd, ld)
cl_c1, cl_c2 = st.columns([1, 1])
with cl_c1:
    fig = chart_cluster_bar(cl_df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
with cl_c2:
    if cl_df is not None:
        disp_cl = cl_df.copy()
        disp_cl["Shipments"] = disp_cl["Shipments"].apply(fmt_num)
        disp_cl["SKUs"] = disp_cl["SKUs"].apply(fmt_num)
        disp_cl["Weight (T)"] = disp_cl["Weight (T)"].apply(lambda x: f"{x:,.1f}")
        disp_cl["Spend (Rs)"] = disp_cl["Spend (Rs)"].apply(fmt_rs)
        disp_cl["Waste (Rs)"] = disp_cl["Waste (Rs)"].apply(fmt_rs)
        disp_cl["Waste %"] = disp_cl["Waste %"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
        disp_cl["Avg Util %"] = disp_cl["Avg Util %"].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")
        disp_cl["Avg CPT"] = disp_cl["Avg CPT"].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "-")
        st.dataframe(disp_cl, use_container_width=True, hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 3. ROUTE x TRUCK TYPE WASTE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">3. Route x Truck Type Waste</div>', unsafe_allow_html=True)

rt_c1, rt_c2 = st.columns([1, 1])
with rt_c1:
    ol = filtered.get("Outlier Legs", ld)
    fig = chart_top_waste_routes(ol)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No route data")
with rt_c2:
    fig = chart_route_truck_heatmap(ld)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No heatmap data")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 4. MONTH-END vs MID + TDP PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">4. Month-End vs Mid & TDP Performance</div>', unsafe_allow_html=True)

me_c1, me_c2 = st.columns([1, 1])
with me_c1:
    result = chart_month_end_vs_mid(lpd, ld)
    if result:
        fig, me_df = result
        st.plotly_chart(fig, use_container_width=True)
        disp_me = me_df.copy()
        disp_me["Shipments"] = disp_me["Shipments"].apply(fmt_num)
        disp_me["SKUs"] = disp_me["SKUs"].apply(fmt_num)
        disp_me["Spend (Rs)"] = disp_me["Spend (Rs)"].apply(fmt_rs)
        disp_me["Waste (Rs)"] = disp_me["Waste (Rs)"].apply(fmt_rs)
        disp_me["Waste %"] = disp_me["Waste %"].apply(lambda x: f"{x:.2f}%")
        disp_me["Avg Util %"] = disp_me["Avg Util %"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(disp_me, use_container_width=True, hide_index=True)
    else:
        st.info("No month-end data")
with me_c2:
    fig = chart_tdp_performance(lpd, ld)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No TDP data")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 5. UTILISATION & TRENDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">5. Utilisation & Trends</div>', unsafe_allow_html=True)

ut_c1, ut_c2 = st.columns([1, 1])
with ut_c1:
    fig = chart_utilisation_dist(ld)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No utilisation data")
with ut_c2:
    fig = chart_monthly_trend(ld)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No monthly data")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 6. PRODUCT-WISE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">6. Product-Wise Insights</div>', unsafe_allow_html=True)

if not lpd.empty and "Material" in lpd.columns:
    merged = lpd.merge(
        products[["Material", "Brand Code Des", "Basepack Desc"]].drop_duplicates("Material"),
        on="Material", how="left",
    )

    brand_summary = merged.groupby("Brand Code Des", dropna=True).agg(
        Materials=("Material", "nunique"), Shipments=("Leg ID", "nunique"),
        Total_Weight=("Net Weight (T)", "sum"), Total_Spend=("Leg Cost (Rs)", "sum"),
        Avg_CPT=("Leg CPT", "mean"), Routes=("Route Code", "nunique"),
    ).reset_index()
    brand_summary.columns = ["Brand", "Materials", "Shipments", "Total Weight (T)",
                              "Total Spend (Rs)", "Avg CPT (Rs)", "Routes"]
    brand_summary = brand_summary.sort_values("Total Spend (Rs)", ascending=False)

    if not ld.empty and "Waste (Rs)" in ld.columns:
        leg_brand = merged[["Leg ID", "Brand Code Des"]].drop_duplicates("Leg ID")
        ld_branded = ld.merge(leg_brand, on="Leg ID", how="left")
        waste_by_brand = ld_branded.groupby("Brand Code Des", dropna=True).agg(
            Total_Waste=("Waste (Rs)", "sum"),
        ).reset_index()
        waste_by_brand.columns = ["Brand", "Total Waste (Rs)"]

        # Weight-share weighted utilisation by brand
        brand_util = weighted_util(merged, ld, group_col="Brand Code Des")
        brand_util = brand_util.reset_index()
        brand_util.columns = ["Brand", "Avg Utilisation %"]

        brand_summary = brand_summary.merge(waste_by_brand, on="Brand", how="left").fillna(0)
        brand_summary = brand_summary.merge(brand_util, on="Brand", how="left").fillna(0)
        brand_summary["Waste %"] = brand_summary["Total Waste (Rs)"] / brand_summary["Total Spend (Rs)"].replace(0, pd.NA) * 100
    else:
        brand_summary["Total Waste (Rs)"] = 0
        brand_summary["Avg Utilisation %"] = 0
        brand_summary["Waste %"] = 0

    # Insight cards
    top_spender = brand_summary.iloc[0] if len(brand_summary) > 0 else None
    top_waster = brand_summary.nlargest(1, "Total Waste (Rs)").iloc[0] if len(brand_summary) > 0 else None
    worst_util = brand_summary.nsmallest(1, "Avg Utilisation %").iloc[0] if len(brand_summary) > 0 and brand_summary["Avg Utilisation %"].sum() > 0 else None

    ic1, ic2, ic3 = st.columns(3)
    if top_spender is not None:
        ic1.markdown(f"""<div class="insight-card">
            <b>Highest Spend Brand</b><br>
            <span style="font-size:1.2rem; font-weight:700">{top_spender['Brand']}</span><br>
            {fmt_rs(top_spender['Total Spend (Rs)'])} across {int(top_spender['Routes'])} routes
        </div>""", unsafe_allow_html=True)
    if top_waster is not None:
        ic2.markdown(f"""<div class="insight-card">
            <b>Highest Waste Brand</b><br>
            <span style="font-size:1.2rem; font-weight:700">{top_waster['Brand']}</span><br>
            {fmt_rs(top_waster['Total Waste (Rs)'])} ({top_waster['Waste %']:.1f}% of spend)
        </div>""", unsafe_allow_html=True)
    if worst_util is not None:
        ic3.markdown(f"""<div class="insight-card">
            <b>Lowest Utilisation Brand</b><br>
            <span style="font-size:1.2rem; font-weight:700">{worst_util['Brand']}</span><br>
            {worst_util['Avg Utilisation %']:.1f}% avg truck utilisation
        </div>""", unsafe_allow_html=True)

    bc1, bc2 = st.columns(2)
    with bc1:
        fig = chart_brand_spend(brand_summary)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with bc2:
        fig = chart_brand_waste(brand_summary)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Brand Summary**")
    display_brand = brand_summary.copy()
    display_brand["Total Weight (T)"] = display_brand["Total Weight (T)"].apply(lambda x: f"{x:,.1f}")
    display_brand["Total Spend (Rs)"] = display_brand["Total Spend (Rs)"].apply(fmt_rs)
    display_brand["Total Waste (Rs)"] = display_brand["Total Waste (Rs)"].apply(fmt_rs)
    display_brand["Avg CPT (Rs)"] = display_brand["Avg CPT (Rs)"].apply(lambda x: f"{x:,.1f}")
    display_brand["Avg Utilisation %"] = display_brand["Avg Utilisation %"].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")
    display_brand["Waste %"] = display_brand["Waste %"].apply(lambda x: f"{x:.2f}%")
    st.dataframe(display_brand, use_container_width=True, hide_index=True)

    with st.expander("Basepack-Level Summary", expanded=False):
        bp_summary = merged.groupby("Basepack Desc", dropna=True).agg(
            Materials=("Material", "nunique"), Shipments=("Shipment No", "nunique"),
            Total_Weight=("Net Weight (T)", "sum"), Total_Spend=("Leg Cost (Rs)", "sum"),
            Avg_CPT=("Leg CPT", "mean"),
        ).reset_index().sort_values("Total_Spend", ascending=False)
        bp_summary.columns = ["Basepack Desc", "Materials", "Shipments",
                              "Total Weight (T)", "Total Spend (Rs)", "Avg CPT (Rs)"]
        display_bp = bp_summary.copy()
        display_bp["Total Weight (T)"] = display_bp["Total Weight (T)"].apply(lambda x: f"{x:,.1f}")
        display_bp["Total Spend (Rs)"] = display_bp["Total Spend (Rs)"].apply(fmt_rs)
        display_bp["Avg CPT (Rs)"] = display_bp["Avg CPT (Rs)"].apply(lambda x: f"{x:,.1f}")
        st.dataframe(display_bp, use_container_width=True, hide_index=True)
else:
    st.info("No product data available for insights")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 7. TOP 5 TRANSPORTERS — TRUCK-WISE PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">7. Top 5 Transporters — Truck-wise Performance</div>', unsafe_allow_html=True)

top_trans_df, top_trans_list = top_transporters_summary(ld, top_n=5)
if top_trans_df is not None and not top_trans_df.empty:
    tr_c1, tr_c2 = st.columns([1, 1])
    with tr_c1:
        st.markdown("**Top 5 Transporters (by Spend)**")
        disp_t = top_trans_df.copy()
        disp_t["Legs"] = disp_t["Legs"].apply(fmt_num)
        disp_t["Weight (T)"] = disp_t["Weight (T)"].apply(lambda x: f"{x:,.1f}")
        disp_t["Spend (Rs)"] = disp_t["Spend (Rs)"].apply(fmt_rs)
        disp_t["Waste (Rs)"] = disp_t["Waste (Rs)"].apply(fmt_rs)
        disp_t["Waste %"] = disp_t["Waste %"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
        disp_t["Avg Util %"] = disp_t["Avg Util %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) and x > 0 else "-")
        disp_t["Avg CPT"] = disp_t["Avg CPT"].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "-")
        st.dataframe(disp_t, use_container_width=True, hide_index=True)
    with tr_c2:
        fig = chart_transporter_truck_mix(ld, top_trans_list)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Truck-wise Breakdown per Transporter**")
    tbd = transporter_truck_breakdown(ld, top_trans_list)
    if tbd is not None and not tbd.empty:
        disp_b = tbd.copy()
        disp_b["Legs"] = disp_b["Legs"].apply(fmt_num)
        disp_b["Weight (T)"] = disp_b["Weight (T)"].apply(lambda x: f"{x:,.1f}")
        disp_b["Spend (Rs)"] = disp_b["Spend (Rs)"].apply(fmt_rs)
        disp_b["Waste (Rs)"] = disp_b["Waste (Rs)"].apply(fmt_rs)
        disp_b["Waste %"] = disp_b["Waste %"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
        disp_b["Avg Util %"] = disp_b["Avg Util %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) and x > 0 else "-")
        disp_b["Avg CPT"] = disp_b["Avg CPT"].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "-")
        st.dataframe(disp_b, use_container_width=True, hide_index=True)
else:
    st.info("No transporter data available for current filters")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 8. DATA TABLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">8. Data Tables</div>', unsafe_allow_html=True)

tab_names = [name for name in SHEET_PARQUETS if name in filtered]
tabs = st.tabs(tab_names)
for tab, name in zip(tabs, tab_names):
    with tab:
        show_sheet_tab(name, filtered[name])
