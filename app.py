"""
Alaska Lightning Detection Network (ALDN) — Streamlit App
Deploy: HuggingFace Spaces
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import calendar
from datetime import datetime, date, timedelta
import os, zipfile, subprocess

# ── Configuración de página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Alaska Lightning Viewer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personalizado ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border-radius: 8px; padding: 12px; border: 1px solid #30363d; }
    .stMetric label { color: #8b949e !important; }
    .stMetric [data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.8rem !important; }
    h1, h2, h3 { color: #e6edf3 !important; }
    .sidebar .sidebar-content { background-color: #161b22; }
    [data-testid="stSidebar"] { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

CURRENT_YEAR = datetime.now().year


# ── Carga y caché de datos ────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Cargando datos ALDN...")
def load_data() -> pd.DataFrame:
    """Descarga datos del ALDN o genera demo si no disponibles."""
    # 1. Intentar cargar archivo local (si se subió al Space)
    for fname in [f"aldn_{CURRENT_YEAR}.parquet", f"aldn_{CURRENT_YEAR}.csv"]:
        if os.path.exists(fname):
            df = pd.read_parquet(fname) if fname.endswith('.parquet') else pd.read_csv(fname)
            df['DATETIME'] = pd.to_datetime(df['DATETIME'])
            return df

    # 2. Intentar descarga directa del ALDN
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
            gdf = gpd.read_file(shp_path)
            gdf = gdf.to_crs('EPSG:4326')
            # Manejar columnas duplicadas DATETIME
            cols = list(gdf.columns)
            new_cols = []
            seen = {}
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            gdf.columns = new_cols
            # Buscar columna de datetime string legible
            dt_col = next((c for c in gdf.columns if 'DATETIME' in c and gdf[c].dtype == object), None)
            if dt_col:
                gdf['DATETIME_FULL'] = pd.to_datetime(gdf[dt_col], format='%Y/%m/%d %H:%M', errors='coerce')
            else:
                dt_col2 = next((c for c in gdf.columns if 'DATETIME' in c), None)
                gdf['DATETIME_FULL'] = pd.to_datetime(gdf[dt_col2], errors='coerce') if dt_col2 else pd.NaT
            df = pd.DataFrame({
                'DATETIME': gdf['DATETIME_FULL'],
                'DATE':     gdf['DATETIME_FULL'].dt.strftime('%Y-%m-%d'),
                'MONTH':    gdf['DATETIME_FULL'].dt.month,
                'HOUR':     gdf['DATETIME_FULL'].dt.hour,
                'LAT':      gdf.geometry.y,
                'LON':      gdf.geometry.x,
                'AMPLITUDE': gdf['AMPLITUDE'] / 1000 if 'AMPLITUDE' in gdf.columns else 0,
                'TYPE':     gdf['TYPE'] if 'TYPE' in gdf.columns else 'UNKNOWN',
                'POLARITY': gdf['POLARITY'] if 'POLARITY' in gdf.columns else 'UNKNOWN',
            })
            return df.dropna(subset=['LAT', 'LON'])
    except Exception:
        pass

    # 3. Dataset sintético de demostración
    np.random.seed(42)
    n = 18000
    months = np.random.choice([5,6,7,8,9], n, p=[0.04, 0.22, 0.42, 0.25, 0.07])
    days = np.random.randint(1, 29, n)
    hours = np.random.choice(range(24), n)
    datetimes = pd.to_datetime(
        {'year': CURRENT_YEAR, 'month': months, 'day': days, 'hour': hours}
    )
    return pd.DataFrame({
        'DATETIME':     datetimes,
        'LAT':          np.random.normal(64.5, 4.2, n).clip(55, 71),
        'LON':          np.random.normal(-153, 8, n).clip(-170, -132),
        'AMPLITUDE':    np.random.normal(-30, 14, n),
        'MULTIPLICITY': np.random.choice([1,2,3,4,5], n, p=[0.5,0.25,0.12,0.08,0.05]),
        'TYPE':         np.random.choice(['GROUND_STROKE', 'CLOUD_STROKE'], n, p=[0.70, 0.30]),
        'POLARITY':     np.random.choice(['Negative', 'Positive', 'Cloud To Cloud'], n, p=[0.60, 0.10, 0.30]),
        'MONTH':        months,
        'HOUR':         hours,
        'DATE':         datetimes.dt.strftime('%Y-%m-%d'),
    })


# ── Cargar datos ──────────────────────────────────────────────────────────
df_full = load_data()
df_full['DATE'] = pd.to_datetime(df_full['DATETIME']).dt.date


# ── Sidebar: Controles ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## ⚡ Alaska Lightning {CURRENT_YEAR}")
    st.markdown("---")

    st.subheader("📅 Rango de Fechas")
    min_date = df_full['DATE'].min()
    max_date = df_full['DATE'].max()
    date_range = st.date_input(
        "Seleccionar rango:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_date, max_date

    st.subheader("🔧 Filtros")
    if 'TYPE' in df_full.columns:
        stroke_types = st.multiselect(
            "Tipo de rayo:",
            options=df_full['TYPE'].unique().tolist(),
            default=df_full['TYPE'].unique().tolist()
        )
    else:
        stroke_types = None

    if 'AMPLITUDE' in df_full.columns:
        amp_range = st.slider(
            "Amplitud (kA):",
            float(df_full['AMPLITUDE'].min()),
            float(df_full['AMPLITUDE'].max()),
            (float(df_full['AMPLITUDE'].min()), float(df_full['AMPLITUDE'].max()))
        )
    else:
        amp_range = None

    max_points = st.slider("Máx. puntos en mapa:", 1000, 20000, 5000, 1000)

    st.markdown("---")
    st.caption("Fuente: [ALDN — BLM Alaska](https://fire.ak.blm.gov/)")


# ── Filtrar datos ─────────────────────────────────────────────────────────
mask = (df_full['DATE'] >= start_d) & (df_full['DATE'] <= end_d)
if stroke_types is not None:
    mask &= df_full['TYPE'].isin(stroke_types)
if amp_range is not None:
    mask &= df_full['AMPLITUDE'].between(amp_range[0], amp_range[1])
df = df_full[mask].copy()


# ── Header ────────────────────────────────────────────────────────────────
st.title("⚡ Alaska Lightning Detection Network")
st.markdown(f"**Período:** {start_d} → {end_d} &nbsp;|&nbsp; "
            f"**Registros:** {len(df):,} rayos detectados")
st.markdown("---")


# ── KPIs ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("⚡ Total Rayos", f"{len(df):,}")
if 'TYPE' in df.columns and len(df) > 0:
    cg = len(df[df['TYPE']=='GROUND_STROKE'])
    ic = len(df[df['TYPE']=='CLOUD_STROKE'])
    col2.metric("↓ GROUND_STROKE", f"{cg:,}", f"{cg/len(df)*100:.0f}%")
    col3.metric("↕ CLOUD_STROKE",  f"{ic:,}", f"{ic/len(df)*100:.0f}%")
if 'AMPLITUDE' in df.columns and len(df) > 0:
    col4.metric("⚡ kA Promedio", f"{df['AMPLITUDE'].abs().mean():.1f}")
if len(df) > 0:
    top_day = df.groupby('DATE').size().idxmax()
    col5.metric("📅 Día más activo", str(top_day))


# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Mapa Espacial", "📊 Dinámicas", "📈 Análisis Temporal", "📋 Datos"])


# ─── Tab 1: Mapa ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Distribución Espacial de Rayos")
    map_type = st.radio("Tipo de visualización:",
                        ["Mapa de Calor", "Puntos (scatter)"], horizontal=True)

    sample = df.sample(min(max_points, len(df)), random_state=42) if len(df) > max_points else df

    m = folium.Map(location=[64.5, -153], zoom_start=4,
                   tiles='CartoDB dark_matter', prefer_canvas=True)

    if map_type == "Mapa de Calor":
        heat_data = sample[['LAT', 'LON']].dropna().values.tolist()
        HeatMap(heat_data, radius=8, blur=6, min_opacity=0.3,
                gradient={0.2:'#0000ff', 0.45:'#00ffff',
                          0.65:'#00ff00', 0.85:'#ffff00', 1.0:'#ff0000'}).add_to(m)
    else:
        for _, row in sample.iterrows():
            folium.CircleMarker(
                location=[row['LAT'], row['LON']],
                radius=2, color='gold', fill=True, fill_opacity=0.5
            ).add_to(m)

    st_folium(m, width=None, height=550, returned_objects=[])

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
        if 'HOUR' in df.columns and len(df) > 0:
            hourly = df.groupby('HOUR').size().reset_index(name='count')
            fig_h = px.area(hourly, x='HOUR', y='count', template='plotly_dark',
                            title="Ciclo Diurno (Hora Local AKST)",
                            labels={'HOUR': 'Hora', 'count': 'Rayos'})
            fig_h.update_traces(line_color='#58a6ff', fillcolor='rgba(88,166,255,0.2)')
            fig_h.update_xaxes(tickmode='array', tickvals=list(range(0,24,3)),
                               ticktext=[f'{h:02d}h' for h in range(0,24,3)])
            st.plotly_chart(fig_h, use_container_width=True)

    with c2:
        if 'TYPE' in df.columns and 'MONTH' in df.columns and len(df) > 0:
            type_month = df.groupby(['MONTH', 'TYPE']).size().reset_index(name='count')
            type_month['MES'] = type_month['MONTH'].map(
                {m: calendar.month_abbr[m] for m in range(1, 13)}
            )
            fig_tm = px.bar(type_month, x='MES', y='count', color='TYPE',
                            template='plotly_dark', barmode='group',
                            title="Tipo de Rayo por Mes",
                            labels={'count': 'Rayos', 'MES': 'Mes'})
            st.plotly_chart(fig_tm, use_container_width=True)

    if 'HOUR' in df.columns and 'MONTH' in df.columns and len(df) > 0:
        pivot = df.pivot_table(index='HOUR', columns='MONTH', aggfunc='size', fill_value=0)
        pivot.columns = [calendar.month_abbr[c] for c in pivot.columns]
        fig_hm = px.imshow(pivot, color_continuous_scale='YlOrRd', template='plotly_dark',
                           title="Heatmap: Hora del Día × Mes",
                           labels={'x': 'Mes', 'y': 'Hora', 'color': 'Rayos'})
        fig_hm.update_layout(height=400)
        st.plotly_chart(fig_hm, use_container_width=True)

    if 'AMPLITUDE' in df.columns and len(df) > 0:
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
            if 'TYPE' in df.columns:
                fig_box = px.box(df, x='TYPE', y='AMPLITUDE', color='TYPE',
                                 template='plotly_dark',
                                 title="Amplitud por Tipo de Rayo",
                                 labels={'TYPE': 'Tipo', 'AMPLITUDE': 'Amplitud (kA)'})
                st.plotly_chart(fig_box, use_container_width=True)


# ─── Tab 3: Análisis Temporal ─────────────────────────────────────────────
with tab3:
    st.subheader("Análisis Temporal")

    if len(df) > 0 and 'DATE' in df.columns:
        daily_counts = df.groupby('DATE').size().reset_index(name='count')
        daily_counts['DATE'] = pd.to_datetime(daily_counts['DATE'])
        daily_counts['rolling7']  = daily_counts['count'].rolling(7,  center=True).mean()
        daily_counts['rolling30'] = daily_counts['count'].rolling(30, center=True).mean()

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Bar(
            x=daily_counts['DATE'], y=daily_counts['count'],
            name='Diario', marker_color='rgba(88,166,255,0.4)'
        ))
        fig_ts.add_trace(go.Scatter(
            x=daily_counts['DATE'], y=daily_counts['rolling7'],
            name='Media 7 días', line=dict(color='#58a6ff', width=2.5)
        ))
        fig_ts.add_trace(go.Scatter(
            x=daily_counts['DATE'], y=daily_counts['rolling30'],
            name='Media 30 días', line=dict(color='#ff7b72', width=2, dash='dash')
        ))
        fig_ts.update_layout(
            template='plotly_dark',
            title=f"Serie Temporal Diaria — Alaska Lightning {CURRENT_YEAR}",
            xaxis_title="Fecha", yaxis_title="Rayos/día",
            height=450, legend=dict(orientation='h', y=1.05)
        )
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
    display_cols = [c for c in df.columns if c != 'geometry']
    st.dataframe(
        df[display_cols].sort_values('DATETIME', ascending=False).head(1000),
        use_container_width=True, height=400
    )
    csv = df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Descargar CSV filtrado",
        data=csv,
        file_name=f"aldn_{CURRENT_YEAR}_{start_d}_{end_d}.csv",
        mime="text/csv"
    )

    st.subheader("📊 Estadísticas del Filtro")
    desc_cols = [c for c in ['LAT', 'LON', 'AMPLITUDE', 'MULTIPLICITY'] if c in df.columns]
    st.dataframe(df[desc_cols].describe().round(2), use_container_width=True)
