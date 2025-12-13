import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import random
import joblib
from PIL import Image
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Executive Ride Delay Dashboard", layout="wide")

# =========================
# CUSTOM CSS
# =========================
custom_css = """
<style>
/* Logo bar */
.logo-bar {
    background-color: #0069AC;
    padding: 10px 20px;
    border-radius: 0px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 20px;
}

/* Card styles */
.big-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    border: 1px solid #e0e0e0;
}

.metric-card {
    background-color: #0069AC;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 5px 8px rgba(0,0,0,0.12);
    border: 3px solid #f8fbff;
    color: #f8fbff
}

.insight-box {
    background-color: #eef6ff;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
    border-left: 8px solid #0069AC;
    font-size: 16px;
}

/* Sidebar filter color */
.css-1pahdxg-control .css-1hwfws3 {
    background-color: #0069AC !important;
    color: white !important;
    border-radius: 5px;
}
.css-1wy0on6 .css-1hwfws3 {
    background-color: #0069AC !important;
    color: white !important;
    border-radius: 5px;
}
.css-1pahdxg-control .css-1r6slb0 {
    background-color: #e6f0fa !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# =========================
# LOGO / HEADER
# =========================
logo = Image.open("logo.png")  
st.image(logo, width=120)  
st.markdown('<div class="logo-bar">🚖 Ride Performance Monitoring Dashboard</div>', unsafe_allow_html=True)


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("model_trained.csv")
kpi = pd.read_csv("KPI_data.csv")
df["delay_label"] = df["delay"].map({0: "No Delay", 1: "Delay"})

# =========================
# LOAD MODEL and ENCODERS
# =========================
model = joblib.load("delay_logreg_model.pkl")
city_encoder = joblib.load("Pickup City_encoder.pkl")
service_encoder = joblib.load("ServiceAreaCode_encoder.pkl")

# =========================
# KPI ANALYSIS 
# =========================


# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("🔍 Filters")

service_filter = st.sidebar.multiselect(
    "Service Area",
    df["ServiceArea"].unique(),
    default=df["ServiceArea"].unique()
)

city_filter = st.sidebar.multiselect(
    "Pickup City",
    df["PickupCity"].unique(),
    default=df["PickupCity"].unique()
)

weekend_filter = st.sidebar.selectbox(
    "Weekend / Weekday",
    ["Both", "Weekend Only", "Weekday Only"]
)

df_f = df[
    df["ServiceArea"].isin(service_filter)
    & df["PickupCity"].isin(city_filter)
]

if weekend_filter == "Weekend Only":
    df_f = df_f[df_f["IsWeekend"] == 1]
elif weekend_filter == "Weekday Only":
    df_f = df_f[df_f["IsWeekend"] == 0]

# =========================
# TABS
# =========================
tab_main, tab_pred, tab_breach, tab_geo = st.tabs(
    ["📊 Overview & Operations", "🤖 Delay Predictions", "🚨 Quality Compliance","🗺 Geo View"]
)

# =========================
# TAB 1 — OVERVIEW
# =========================
with tab_main:
   
    # -------------------------
    # Executive insights
    # -------------------------
    wk_delay = df_f[df_f["IsWeekend"] == 0]["delay"].mean()
    we_delay = df_f[df_f["IsWeekend"] == 1]["delay"].mean()
    service_delay = df_f.groupby('ServiceArea')['delay'].mean()
    top_service = service_delay.idxmax()
    city_delay = df_f.groupby('PickupCity')['delay'].sum()
    top_city_delay = city_delay.idxmax()
    hourly_delay = df_f.groupby('Pickup_Hour')['delay'].mean()
    peak_hour = hourly_delay.idxmax()
    top_city_rides = df_f['PickupCity'].value_counts().idxmax()
    total_delays = df_f['delay'].sum()
    avg_dist_delay = df_f[df_f['delay'] == 1]['Distance'].mean()
    avg_dist_all = df_f['Distance'].mean()
    busiest_hour = df_f['Pickup_Hour'].value_counts().idxmax()

    insights = [
    f"📌 Delays are <b>{abs(wk_delay - we_delay)*100:.1f}%</b> "
    + ("more common" if we_delay > wk_delay else "less common")
    + " on weekdays than weekends.",

    f"⏰ Busiest pickup hour: <b>{busiest_hour}:00</b>",

    f"📌 Service area with highest delay rate: <b>{top_service}</b> "
    f"(<b>{service_delay[top_service]*100:.1f}%</b>)",

    f"🏙 City with most delayed rides: <b>{top_city_delay}</b> "
    f"(<b>{city_delay[top_city_delay]:,}</b> delays)",

    f"⏰ Hour with highest delay rate: <b>{peak_hour}:00</b> "
    f"(<b>{hourly_delay[peak_hour]*100:.1f}%</b>)",

    f"🏙 Pickup city with most rides: <b>{top_city_rides}</b> "
    f"(<b>{df_f['PickupCity'].value_counts()[top_city_rides]:,}</b> rides)",

    f"📌 Total delayed rides: <b>{total_delays:,}</b> "
    f"(<b>{total_delays/len(df_f)*100:.1f}%</b>)",

    f"📌 Delayed trips average <b>{avg_dist_delay:.1f} mi</b> "
    f"vs <b>{avg_dist_all:.1f} mi</b> for all trips — "
    + ("slightly shorter." if avg_dist_delay < avg_dist_all else "slightly longer.")
]

    cols = st.columns(2)
    for i, insight in enumerate(insights):
        with cols[i % 2]:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="margin-top:20px;">', unsafe_allow_html=True)

    # -------------------------
    # Charts in 3 columns
    # -------------------------
    col1_chart, col2_chart, col3_chart = st.columns(3)

# 1️⃣ Doughnut chart — Delays vs No Delays
    delay_counts = df_f['delay_label'].value_counts()
    delay_percent = df_f['delay_label'].value_counts(normalize=True) * 100
    with col1_chart:
        with st.container():
    
            fig_doughnut = go.Figure(data=[go.Pie(
                labels=delay_counts.index,
                values=delay_counts.values,
                hole=0.5,
                hoverinfo="label+percent+value",
                text=[f"{val} ({percent:.1f}%)" for val, percent in zip(delay_counts.values, delay_percent)],
                textinfo="text",
                marker_colors=["#DC3545","#28A745"]
            )])
            fig_doughnut.update_layout(
                title_text="🚖 Delays vs No Delays (Overall)",
                paper_bgcolor="white",
                plot_bgcolor="white",
                transition={'duration': 1000},
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_doughnut, use_container_width=True)
    
    
    # 2️⃣ Bar Chart — % of Delays by Service Area
    service_area_counts = df_f.groupby(['ServiceArea', 'delay_label']).size().reset_index(name='count')
    service_area_totals = service_area_counts.groupby('ServiceArea')['count'].sum().reset_index(name='total')
    service_area_counts = service_area_counts.merge(service_area_totals, on='ServiceArea')
    service_area_counts['percent'] = service_area_counts['count'] / service_area_counts['total'] * 100
    service_area_counts['label'] = service_area_counts.apply(lambda x: f"{x['count']} ({x['percent']:.1f}%)", axis=1)
    
    with col2_chart:
        with st.container():
            fig_service_area = px.bar(
                service_area_counts,
                x='ServiceArea',
                y='percent',
                color='delay_label',
                text='label',
                barmode='stack',
                color_discrete_map={"No Delay": "#28A745", "Delay": "#DC3545"},
                labels={'percent': '% of Rides', 'ServiceArea': 'Service Area'},
                height=400,
                animation_frame=None
            )
            fig_service_area.update_traces(textposition='inside')
            fig_service_area.update_layout(
                title_text="📊 % and Count of Delays by Service Area",
                paper_bgcolor="white",
                plot_bgcolor="white",
                transition={'duration': 1000},
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_service_area, use_container_width=True)
    

    
    
    # 3️⃣ Column Chart — Top 20 Pickup Cities
    city_counts = df_f.groupby(['PickupCity', 'delay_label']).size().reset_index(name='count')
    city_totals = city_counts.groupby('PickupCity')['count'].sum().reset_index(name='total')
    top_cities = city_totals.sort_values('total', ascending=False).head(20)['PickupCity']
    city_counts = city_counts[city_counts['PickupCity'].isin(top_cities)]
    city_counts = city_counts.merge(city_totals, on='PickupCity')
    city_counts['percent'] = city_counts['count'] / city_counts['total'] * 100
    city_counts['label'] = city_counts.apply(lambda x: f"{x['count']} ({x['percent']:.1f}%)", axis=1)
    
    with col3_chart:
        with st.container():
    
            fig_city = px.bar(
                city_counts,
                y='PickupCity',
                x='percent',
                color='delay_label',
                text='label',
                barmode='stack',
                color_discrete_map={"No Delay": "#28A745", "Delay": "#DC3545"},
                labels={'percent': '% of Rides', 'PickupCity': 'Pickup City'},
                height=600,
                animation_frame=None
            )
            fig_city.update_traces(textposition='outside')
            fig_city.update_layout(
                title_text="🏙 % and Count of Delays by Top 20 Pickup Cities",
                paper_bgcolor="white",
                plot_bgcolor="white",
                transition={'duration': 1000},
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_city, use_container_width=True)
    

    
    # 4️⃣ Full-width — Delays by Hour
    st.markdown('<div style="margin-top:20px;">', unsafe_allow_html=True)  # Add spacing before full-width card
    with st.container():
    
        st.markdown("### ⏰ Delays by Hour of Day")
        df_hour = df_f.groupby(['Pickup_Hour', 'delay_label']).size().reset_index(name='count')
        df_hour['Pickup_Hour_str'] = df_hour['Pickup_Hour'].apply(lambda x: f"{x:02d}:00")
    
        # Calculate percentage per hour
        df_total_hour = df_hour.groupby('Pickup_Hour')['count'].sum().reset_index(name='total')
        df_hour = df_hour.merge(df_total_hour, on='Pickup_Hour')
        df_hour['percent'] = df_hour['count'] / df_hour['total'] * 100
    
        # Create text labels combining count and percent
        df_hour['label'] = df_hour.apply(lambda x: f"{x['count']} ({x['percent']:.1f}%)", axis=1)
    
        fig_hour = px.bar(
            df_hour,
            x='Pickup_Hour_str',
            y='count',
            color='delay_label',
            text='label',
            barmode='stack',
            color_discrete_map={"No Delay": "#28A745", "Delay": "#DC3545"},
            height=400,
            width=800,
            animation_frame=None
        )
        fig_hour.update_traces(textposition='inside')  # Text inside bars
        fig_hour.update_layout(
            xaxis_title="Pickup Hour",
            yaxis_title="Number of Rides",
            paper_bgcolor="white",
            plot_bgcolor="white",
            transition={'duration': 1000},
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_hour, use_container_width=True)



# =========================
# TAB 2 — PREDICTIONS
# =========================
with tab_pred:
    st.subheader("🤖 Predict Ride Delay (Using Logistic Regression Model)")

    col1, col2, col3 = st.columns(3)

    with col1:
        dist = st.number_input("Distance (miles)", min_value=0.1, max_value=100.0)
        base_charge = st.number_input("Base Charge Total ($)", min_value=0.0, max_value=500.0)
        pickup_city = st.selectbox("Pickup City", df["PickupCity"].unique())

    with col2:
        hour = st.slider("Pickup Hour", 0, 23)
        billed_hours = st.number_input("Billed Number Minutes (in hours)", min_value=0.0, max_value=10.0)
        service_area = st.selectbox("Service Area", df["ServiceArea"].unique())

    with col3:
        wkd = st.selectbox("Weekend?", ["Yes", "No"])
        pickup_weekday = st.selectbox("Pickup Weekday (0=Mon,..6=Sun)", list(range(7)))

    # Encode categorical inputs
    pickup_city_enc = city_encoder.transform([str(pickup_city)])[0] if str(pickup_city) in city_encoder.classes_ else 0
    servicearea_code_enc = service_encoder.transform([str(service_area)])[0] if str(service_area) in service_encoder.classes_ else 0

    model_cols = ['Pickup_Hour', 'Pickup_Weekday', 'IsWeekend', 'Distance',
                  'BilledNumberMinutesInHours', 'BaseChargeTotal',
                  'ServiceAreaCode_enc', 'Pickup City_enc']

    sample = pd.DataFrame([{
        "Pickup_Hour": hour,
        "Pickup_Weekday": pickup_weekday,
        "IsWeekend": 1 if wkd == "Yes" else 0,
        "Distance": dist,
        "BilledNumberMinutesInHours": billed_hours,
        "BaseChargeTotal": base_charge,
        "ServiceAreaCode_enc": servicearea_code_enc,
        "Pickup City_enc": pickup_city_enc
    }])

    sample = sample[model_cols]

    if st.button("Predict Delay"):
        pred_prob = model.predict_proba(sample)[0][1]
        pred = model.predict(sample)[0]

        st.metric("Predicted Delay Probability", f"{pred_prob*100:.1f}%")
        st.write("Predicted Class:", "Delay" if pred == 1 else "No Delay")



with tab_breach:

    st.subheader("🚨 Quality & Compliance Overview")

    # ------------------------
    # SUMMARY KPI TABLE
    # ------------------------
    total_trips = len(kpi)

    summary_kpi = pd.DataFrame({
        'Metric': [
            'Total Trips',
            'Trips with Pickup Delay Breach',
            'Trips with Pricing Breach',
            'Trips with Service Acknowledgement Issue',
            'Trips with ANY Red Alert'
        ],
        'Count': [
            total_trips,
            (kpi['PickupDelay_Flag'] == 'DELAY BREACH').sum(),
            (kpi['Pricing_Flag'] == 'PRICING BREACH').sum(),
            kpi['ServiceAck_Flag'].isin(['UNASSIGNED DRIVER', 'OPEN SEGMENT']).sum(),
            kpi['AlertLevel'].str.startswith('RED').sum()
        ]
    })

    summary_kpi['% of Trips'] = (summary_kpi['Count'] / total_trips * 100).round(2)

    def insight_card(title, value, subtitle, icon, color):
        st.markdown(
        f"""
        <div style="
            background: white;
            padding: 18px 22px;
            border-radius: 14px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.10);
            border-left: 6px solid {color};
            margin-bottom: 12px;
        ">
            <div style="font-size: 30px; float:right; margin-top:-5px;">{icon}</div>
            <h3 style="margin:0; font-size: 18px; color:#444;">{title}</h3>
            <p style="margin:5px 0; font-size: 32px; font-weight:700; color:#000;">{value:,}</p>
            <p style="margin:0; font-size: 14px; color:#666;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    

# ==============================
# Compute percentages
# ==============================
    summary_kpi["Pct"] = (summary_kpi["Count"] / summary_kpi.loc[0, "Count"] * 100).round(2)


# ==============================
# Display insight cards
# ==============================

    col1, col2, col3 = st.columns(3)
    
    with col1:
        insight_card(
            summary_kpi.loc[0, "Metric"],
            summary_kpi.loc[0, "Count"],
            "",         # no percentage for total trips
            "🚗",
            "#4e79a7"
        )
    
    with col2:
        insight_card(
            summary_kpi.loc[1, "Metric"],
            summary_kpi.loc[1, "Count"],
            f"{summary_kpi.loc[1, 'Pct']}% of all trips",
            "⏱️",
            "#f28e2b"
        )
    
    with col3:
        insight_card(
            summary_kpi.loc[2, "Metric"],
            summary_kpi.loc[2, "Count"],
            f"{summary_kpi.loc[2, 'Pct']}% of all trips",
            "💲",
            "#e15759"
        )
    
    col4, col5 = st.columns(2)
    
    with col4:
        insight_card(
            summary_kpi.loc[3, "Metric"],
            summary_kpi.loc[3, "Count"],
            f"{summary_kpi.loc[3, 'Pct']}% of all trips",
            "📢",
            "#76b7b2"
        )
    
    with col5:
        insight_card(
            summary_kpi.loc[4, "Metric"],
            summary_kpi.loc[4, "Count"],
            f"{summary_kpi.loc[4, 'Pct']}% of all trips",
            "🛑",
            "#d62728"
        )
    st.markdown("<br>", unsafe_allow_html=True)


    # ============================================
    # ROW 1 – 3 COLUMNS
    # ============================================
    c1, c2, c3 = st.columns(3)

    # ==========================
    # 1. Delay Flag Chart
    # ==========================
    with c1:
        delay_counts = kpi['PickupDelay_Flag'].value_counts().reset_index()
        delay_counts.columns = ['PickupDelay_Flag', 'count']
        delay_counts['percent'] = (delay_counts['count'] / total_trips * 100).round(1)
        delay_counts['label'] = delay_counts.apply(lambda x: f"{x['count']} ({x['percent']}%)", axis=1)

        fig1 = px.bar(
            delay_counts,
            x='PickupDelay_Flag',
            y='count',
            text='label',
            color='PickupDelay_Flag',
            color_discrete_sequence=["#DC3545","#28A745","#FF704D",'#0069AC'],
            labels={'count': 'Number of Rides', 'PickupDelay_Flag': 'Delay Flags'},
            title="🚖 Pickup Delay Flags (%)"
        )
        fig1.update_traces(textposition='auto')
        st.plotly_chart(fig1, use_container_width=True)


    # ==========================
    # 2. Service Acknowledgement
    # ==========================
    with c2:
        ack_counts = kpi['ServiceAck_Flag'].value_counts().reset_index()
        ack_counts.columns = ['ServiceAck_Flag', 'count']
        ack_counts['percent'] = (ack_counts['count'] / total_trips * 100).round(1)
        ack_counts['label'] = ack_counts.apply(lambda x: f"{x['count']} ({x['percent']}%)", axis=1)

        fig2 = px.bar(
            ack_counts,
            x='ServiceAck_Flag',
            y='count',
            text='label',
            color='ServiceAck_Flag',
            color_discrete_sequence=["#28A745","#FF704D",'#0069AC',"#DC3545"],
            labels={'count': 'Number of Rides', 'ServiceAck_Flag': 'Service Acknowledge Flags'},
            title="📝 Service Acknowledgement Issue (%)"
        )
        fig2.update_traces(textposition='auto')
        fig2.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
       

    # ==========================
    # 3. Alert Levels
    # ==========================
    with c3:
        alert_counts = kpi['AlertLevel'].value_counts().reset_index()
        alert_counts.columns = ['AlertLevel', 'count']
        alert_counts['percent'] = (alert_counts['count'] / total_trips * 100).round(1)
        alert_counts['label'] = alert_counts.apply(lambda x: f"{x['count']} ({x['percent']}%)", axis=1)

      
        fig3 = px.bar(
            alert_counts,
            x='AlertLevel',
            y='count',
            text='label',
            color='AlertLevel',
            labels={'count': 'Number of Rides', 'AlertLevel': "Alert Level"},
            title="🚨 Alerts (%)",
            color_discrete_sequence=["#DC3545", "#FF704D", "#28A745","#CC0000"]
        )
        fig3.update_traces(textposition='auto')
        fig3.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)
       


    # ============================================
    # ROW 2 – HEATMAP + STACKED BAR
    # ============================================
    d1, d2 = st.columns([1,1])
    with d1:
        kpi['Pickup_Hour'] = pd.to_datetime(kpi['Pickup DateTime']).dt.hour
        kpi['Weekday'] = pd.to_datetime(kpi['Pickup DateTime']).dt.day_name()
        kpi['IsRed'] = kpi['AlertLevel'].str.startswith("RED")
        heat = kpi.groupby(['Weekday', 'Pickup_Hour'])['IsRed'].mean().reset_index()
        heat['Pct'] = (heat['IsRed'] * 100).round(1)
        
        # Re-order weekdays
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heat = heat.pivot(index="Weekday", columns="Pickup_Hour", values="Pct").reindex(order)
        
        # --------------------------
        # Use imshow to allow cell text
        # --------------------------
        fig4 = px.imshow(
            heat,
            text_auto=True,   # <-- Show % inside each cell
            aspect="auto",
            color_continuous_scale=['#d4f4dd', '#a8e6a3', '#7cd66a', '#ffb366', '#ff704d'],
            labels=dict(x="Pickup Hour", y="Day of Week", color="% Red Alerts"),
            title="🔥 Heatmap: Red Alert % by Hour & Weekday"
        )
        
        # Show discrete hourly ticks (00–23)
        fig4.update_xaxes(
            tickmode='array',
            tickvals=list(range(24)),
            ticktext=[f"{i:02d}:00" for i in range(24)]
        )
        
        # Bold titles and improve layout
        fig4.update_layout(
            title_font=dict(size=20, color="black"),
            xaxis_title="Pickup Hour",
            yaxis_title="Day of Week",
            coloraxis_colorbar=dict(title="% Red Alerts"),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        st.plotly_chart(fig4, use_container_width=True)

        
    # ==========================
    # 5. Stacked Bar (Service Area)
    # ==========================
    with d2:
        alert_area = kpi.groupby(['ServiceAreaCode', 'AlertLevel']).size().reset_index(name='count')
        alert_area['percent'] = alert_area.groupby('ServiceAreaCode')['count'].transform(lambda x: x / x.sum() * 100)       
        alert_area['label'] = alert_area.apply(lambda x: f"{x['count']} ({x['percent']:.1f}%)", axis=1)

        fig5 = px.bar(
            alert_area,
            x='ServiceAreaCode',
            y='count',
            color='AlertLevel',
            labels={'count': 'Number of Rides', 'ServiceAreaCode': "Service Area Code"},
            text='label',
            barmode='stack',
            title="📦 Alert Breakdown by Service Area (%)",
            color_discrete_sequence=["#28A745","#DC3545","#FF704D","#DC3545"]
        )
        fig5.update_traces(textposition='inside')
        st.plotly_chart(fig5, use_container_width=True)


# =========================
# TAB 3 — GEO
# =========================
with tab_geo:
    st.subheader("🗺 Pickup City Performance Map (Delay Rate by City)")

    # --- Coordinates ---
    city_coords = {
        "New York": (40.7128, -74.0060),
        "JFK": (40.6413, -73.7781),
        "DCA": (38.8512, -77.0402),
        "IAD": (38.9531, -77.4565),
        "Uniondale": (40.7004, -73.5929),
        "Westbury": (40.7557, -73.5876),
        "Beverly Hills": (34.0736, -118.4004),
        "Los Angeles": (34.0522, -118.2437),
    }

    def coord(city):
        return city_coords.get(city, (None, None))

    # Copy & generate unique TripID
    df_map = df_f.copy().reset_index().rename(columns={"index": "TripID"})

    df_map["lat"] = df_map["PickupCity"].apply(lambda x: coord(x)[0])
    df_map["lon"] = df_map["PickupCity"].apply(lambda x: coord(x)[1])

    # --- Aggregate by city ---
    city_summary = df_map.groupby(["PickupCity", "lat", "lon"]).agg(
        total_trips=("TripID", "count"),
        delay_trips=("delay_label", lambda x: (x == "Delay").sum())
    ).reset_index()

    city_summary["delay_rate"] = (
        city_summary["delay_trips"] / city_summary["total_trips"] * 100
    ).round(1)

    city_summary["size"] = city_summary["total_trips"] * 3  # bubble scaling

    fig = px.scatter_mapbox(
        city_summary,
        lat="lat",
        lon="lon",
        size="size",
        color="delay_rate",
        hover_name="PickupCity",
        hover_data={
            "total_trips": True,
            "delay_trips": True,
            "delay_rate": True,
            "lat": False,
            "lon": False
        },
        color_continuous_scale=["#28A745", "#FFC107", "#DC3545"],
        zoom=4,
        height=600
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        coloraxis_colorbar=dict(title="% Delay Rate"),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)
