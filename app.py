"""
Alaska Lightning Detection Network (ALDN) — Streamlit App
Deploy: HuggingFace Spaces
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import calendar
from datetime import datetime, date
import os, zipfile

# ── Configuración de página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Alaska Lightning Viewer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMetric { background-color: #161b22; border-radius: 8px; padding: 12px; border: 1px solid #30363d; }
    .stMetric label { color: #8b949e !important; }
    .stMetric [data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.8rem !important; }
    [data-testid="stSidebar"] { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

CURRENT_YEAR = datetime.now().year


@st.cache_data(ttl=3600, show_spinner="Cargando datos ALDN...")
def load_data() -> pd.DataFrame:
    # 1. Archivo local
    for fname in [f"aldn_{CURRENT_YEAR}.parquet", f"aldn_{CURRENT_YEAR}.csv"]:
        if os.path.exists(fname):
            df = pd.read_parquet(fname) if fname.endswith('.parquet') else pd.read_csv(fname)
            df['DATETIME'] = pd.to_datetime(df['DATETIME'])
            return df

    # 2. Descarga directa ALDN
    try:
        url = "https://fire.ak.blm.gov/content/maps/aicc/Data/Data%20(zipped%20Shapefiles)/CurrentYearLightning_SHP.zip"
        import urllib.request
        urllib.request.urlretrieve(url, "lightning.zip")
        with zipfile.ZipFile("lightning.zip") as z:
            z.extractall("lightning_data")
        import geopandas as gpd
        shp_path = None
        for root, _, files in os.walk("lightning_data"):
            for f in files:
                if f.endswith(".shp"):
                    shp_path = os.path.join(root, f)
        if shp_path:
            gdf = gpd.read_file(shp_path).to_crs('EPSG:4326')
            cols = list(gdf.columns)
            seen = {}
            new_cols = []
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            gdf.columns = new_cols
            dt_col = next((c for c in gdf.columns if 'DATETIME' in c and gdf[c].dtype == object), None)
            gdf['DATETIME_FULL'] = pd.to_datetime(gdf[dt_col], format='%Y/%m/%d %H:%M', errors='coerce') if dt_col else pd.NaT
            return pd.DataFrame({
                'DATETIME':  gdf['DATETIME_FULL'],
                'DATE':      gdf['DATETIME_FULL'].dt.strftime('%Y-%m-%d'),
                'MONTH':     gdf['DATETIME_FULL'].dt.month,
                'HOUR':      gdf['DATETIME_FULL'].dt.hour,
                'LAT':       gdf.geometry.y,
                'LON':       gdf.geometry.x,
                'AMPLITUDE': gdf['AMPLITUDE'] / 1000 if 'AMPLITUDE' in gdf.columns else 0,
                'TYPE':      gdf['TYPE'] if 'TYPE' in gdf.columns else 'UNKNOWN',
                'POLARITY':  gdf['POLARITY'] if 'POLARITY' in gdf.columns else 'UNKNOWN',
            }).dropna(subset=['LAT', 'LON'])
    except Exception:
        pass

    # 3. Dataset demo
    np.random.seed(42)
    n = 18000
    months = np.random.choice([5,6,7,8,9], n, p=[0.04, 0.22, 0.42, 0.25, 0.07])
    days   = np.random.randint(1, 29, n)
    hours  = np.random.choice(range(24), n)
    datetimes = pd.to_datetime({'year': CURRENT_YEAR, 'month': months, 'day': days, 'hour': hours})
    return pd.DataFrame({
        'DATETIME':  datetimes,
        'LAT':       np.random.normal(64.5, 4.2, n).clip(55, 71),
        'LON':       np.random.normal(-153, 8, n).clip(-170, -132),
        'AMPLITUDE': np.random.normal(-30, 14, n),
        'TYPE':      np.random.choice(['GROUND_STROKE', 'CLOUD_STROKE'], n, p=[0.70, 0.30]),
        'POLARITY':  np.random.choice(['Negative', 'Positive', 'Cloud To Cloud'], n, p=[0.60, 0.10, 0.30]),
        'MONTH':     months,
        'HOUR':      hours,
        'DATE':      datetimes.dt.strftime('%Y-%m-%d'),
    })


# ── Cargar datos ──────────────────────────────────────────────────────────
df_full = load_data()
df_full['DATE'] = pd.to_datetime(df_full['DATETIME']).dt.date

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## ⚡ Alaska Lightning {CURRENT_YEAR}")
    st.markdown("---")
    st.subheader("📅 Rango de Fechas")
    min_date = df_full['DATE'].min()
    max_date = df_full['DATE'].max()
    date_range = st.date_input("Seleccionar rango:",
                               value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)
    start_d, end_d = date_range if len(date_range) == 2 else (min_date, max_date)

    st.subheader("🔧 Filtros")
    stroke_types = st.multiselect("Tipo de rayo:",
                                  options=df_full['TYPE'].unique().tolist(),
                                  default=df_full['TYPE'].unique().tolist())
    amp_range = st.slider("Amplitud (kA):",
                          float(df_full['AMPLITUDE'].min()),
                          float(df_full['AMPLITUDE'].max()),
                          (float(df_full['AMPLITUDE'].min()), float(df_full['AMPLITUDE'].max())))
    max_points = st.slider("Máx. puntos en mapa:", 1000, 20000, 5000, 1000)
    st.markdown("---")
    st.caption("Fuente: [ALDN — BLM Alaska](https://fire.ak.blm.gov/)")

# ── Filtrar ───────────────────────────────────────────────────────────────
mask = ((df_full['DATE'] >= start_d) & (df_full['DATE'] <= end_d) &
        df_full['TYPE'].isin(stroke_types) &
        df_full['AMPLITUDE'].between(amp_range[0], amp_range[1]))
df = df_full[mask].copy()

# ── Header ────────────────────────────────────────────────────────────────
st.title("⚡ Alaska Lightning Detection Network")
st.markdown(f"**Período:** {start_d} → {end_d} &nbsp;|&nbsp; **Registros:** {len(df):,} rayos detectados")
st.markdown("---")

# ── KPIs ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("⚡ Total Rayos", f"{len(df):,}")
if len(df) > 0:
    cg = len(df[df['TYPE']=='GROUND_STROKE'])
    ic = len(df[df['TYPE']=='CLOUD_STROKE'])
    col2.metric("↓ GROUND_STROKE", f"{cg:,}", f"{cg/len(df)*100:.0f}%")
    col3.metric("↕ CLOUD_STROKE",  f"{ic:,}", f"{ic/len(df)*100:.0f}%")
    col4.metric("⚡ kA Promedio", f"{df['AMPLITUDE'].abs().mean():.1f}")
    col5.metric("📅 Día más activo", str(df.groupby('DATE').size().idxmax()))

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Mapa Espacial", "📊 Dinámicas", "📈 Análisis Temporal", "📋 Datos"])

# ─── Tab 1: Mapa ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Distribución Espacial de Rayos")
    map_type = st.radio("Tipo de visualización:",
                        ["Scatter por tipo", "Densidad (hexbin)"], horizontal=True)
    sample = df.sample(min(max_points, len(df)), random_state=42) if len(df) > max_points else df

    if map_type == "Scatter por tipo":
        fig_map = px.scatter_mapbox(
            sample, lat='LAT', lon='LON', color='TYPE',
            zoom=3, center={'lat': 64, 'lon': -153},
            mapbox_style='carto-darkmatter', opacity=0.6,
            color_discrete_map={'GROUND_STROKE': '#ff7b72', 'CLOUD_STROKE': '#79c0ff'},
            title=f'Lightning {CURRENT_YEAR} — {len(sample):,} rayos', height=550
        )
    else:
        fig_map = px.density_mapbox(
            sample, lat='LAT', lon='LON', radius=8,
            zoom=3, center={'lat': 64, 'lon': -153},
            mapbox_style='carto-darkmatter',
            color_continuous_scale='YlOrRd',
            title=f'Densidad de Rayos {CURRENT_YEAR}', height=550
        )
    fig_map.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Distribución por Zona Geográfica")
    df['ZONE'] = pd.cut(df['LON'], bins=[-172, -160, -150, -140, -130],
                        labels=['Oeste Extremo', 'Alaska Oeste', 'Alaska Centro', 'Alaska Este'])
    zone_stats = df.groupby('ZONE', observed=True).size().reset_index(name='count')
    fig_zone = px.bar(zone_stats, x='ZONE', y='count', color='count',
                      color_continuous_scale='Blues', template='plotly_dark',
                      labels={'count': 'Rayos', 'ZONE': 'Zona'},
                      title="Rayos por Zona Geográfica")
    fig_zone.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_zone, use_container_width=True)

# ─── Tab 2: Dinámicas ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Dinámicas de Rayos")
    c1, c2 = st.columns(2)

    with c1:
        if len(df) > 0:
            hourly = df.groupby('HOUR').size().reset_index(name='count')
            fig_h = px.area(hourly, x='HOUR', y='count', template='plotly_dark',
                            title="Ciclo Diurno (Hora Local AKST)",
                            labels={'HOUR': 'Hora', 'count': 'Rayos'})
            fig_h.update_traces(line_color='#58a6ff', fillcolor='rgba(88,166,255,0.2)')
            fig_h.update_xaxes(tickmode='array', tickvals=list(range(0,24,3)),
                               ticktext=[f'{h:02d}h' for h in range(0,24,3)])
            st.plotly_chart(fig_h, use_container_width=True)

    with c2:
        if len(df) > 0:
            type_month = df.groupby(['MONTH', 'TYPE']).size().reset_index(name='count')
            type_month['MES'] = type_month['MONTH'].map({m: calendar.month_abbr[m] for m in range(1,13)})
            fig_tm = px.bar(type_month, x='MES', y='count', color='TYPE',
                            template='plotly_dark', barmode='group',
                            title="Tipo de Rayo por Mes",
                            color_discrete_map={'GROUND_STROKE':'#ff7b72','CLOUD_STROKE':'#79c0ff'},
                            labels={'count': 'Rayos', 'MES': 'Mes'})
            st.plotly_chart(fig_tm, use_container_width=True)

    if len(df) > 0:
        pivot = df.pivot_table(index='HOUR', columns='MONTH', aggfunc='size', fill_value=0)
        pivot.columns = [calendar.month_abbr[c] for c in pivot.columns]
        fig_hm = px.imshow(pivot, color_continuous_scale='YlOrRd', template='plotly_dark',
                           title="Heatmap: Hora del Día × Mes",
                           labels={'x': 'Mes', 'y': 'Hora', 'color': 'Rayos'})
        fig_hm.update_layout(height=400)
        st.plotly_chart(fig_hm, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig_amp = px.histogram(df, x='AMPLITUDE', nbins=60, template='plotly_dark',
                                   title="Distribución de Amplitud (kA)",
                                   color_discrete_sequence=['#3fb950'],
                                   labels={'AMPLITUDE': 'Amplitud (kA)'})
            fig_amp.add_vline(x=df['AMPLITUDE'].mean(), line_dash='dash',
                              line_color='red', annotation_text="Media")
            st.plotly_chart(fig_amp, use_container_width=True)
        with c4:
            fig_box = px.box(df, x='TYPE', y='AMPLITUDE', color='TYPE',
                             template='plotly_dark', title="Amplitud por Tipo de Rayo",
                             color_discrete_map={'GROUND_STROKE':'#ff7b72','CLOUD_STROKE':'#79c0ff'},
                             labels={'TYPE': 'Tipo', 'AMPLITUDE': 'Amplitud (kA)'})
            st.plotly_chart(fig_box, use_container_width=True)

# ─── Tab 3: Análisis Temporal ─────────────────────────────────────────────
with tab3:
    st.subheader("Análisis Temporal")
    if len(df) > 0:
        daily_counts = df.groupby('DATE').size().reset_index(name='count')
        daily_counts['DATE']      = pd.to_datetime(daily_counts['DATE'])
        daily_counts['rolling7']  = daily_counts['count'].rolling(7,  center=True).mean()
        daily_counts['rolling30'] = daily_counts['count'].rolling(30, center=True).mean()

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Bar(x=daily_counts['DATE'], y=daily_counts['count'],
                                name='Diario', marker_color='rgba(88,166,255,0.4)'))
        fig_ts.add_trace(go.Scatter(x=daily_counts['DATE'], y=daily_counts['rolling7'],
                                    name='Media 7 días', line=dict(color='#58a6ff', width=2.5)))
        fig_ts.add_trace(go.Scatter(x=daily_counts['DATE'], y=daily_counts['rolling30'],
                                    name='Media 30 días', line=dict(color='#ff7b72', width=2, dash='dash')))
        fig_ts.update_layout(template='plotly_dark',
                             title=f"Serie Temporal Diaria — Alaska Lightning {CURRENT_YEAR}",
                             xaxis_title="Fecha", yaxis_title="Rayos/día",
                             height=450, legend=dict(orientation='h', y=1.05))
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("🏆 Top 10 Días con Mayor Actividad")
        top10 = df.groupby('DATE').size().nlargest(10).reset_index(name='Rayos')
        top10['Fecha'] = top10['DATE'].astype(str)
        fig_top = px.bar(top10, x='Rayos', y='Fecha', orientation='h',
                         template='plotly_dark', color='Rayos',
                         color_continuous_scale='Plasma')
        fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig_top, use_container_width=True)

        df['WEEK'] = pd.to_datetime(df['DATETIME']).dt.isocalendar().week
        weekly = df.groupby('WEEK').size().reset_index(name='count')
        fig_wk = px.bar(weekly, x='WEEK', y='count', template='plotly_dark',
                        title="Rayos por Semana del Año",
                        labels={'WEEK': 'Semana', 'count': 'Rayos'},
                        color='count', color_continuous_scale='Blues')
        st.plotly_chart(fig_wk, use_container_width=True)

# ─── Tab 4: Datos ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("📋 Datos Filtrados")
    display_cols = [c for c in df.columns if c not in ['geometry', 'ZONE', 'WEEK']]
    st.dataframe(df[display_cols].sort_values('DATETIME', ascending=False).head(1000),
                 use_container_width=True, height=400)
    csv = df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Descargar CSV filtrado", data=csv,
                       file_name=f"aldn_{CURRENT_YEAR}_{start_d}_{end_d}.csv",
                       mime="text/csv")
    st.subheader("📊 Estadísticas del Filtro")
    desc_cols = [c for c in ['LAT', 'LON', 'AMPLITUDE'] if c in df.columns]
    st.dataframe(df[desc_cols].describe().round(2), use_container_width=True)
