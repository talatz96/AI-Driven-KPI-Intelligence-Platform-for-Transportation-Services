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
    # Top Metrics
    # -------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_rides = len(df_f)
        st.markdown(f"""
        <div class="metric-card">
            <p style="font-size: 20px; color: #EAF6F9; text-align: center">Total Rides</p>
            <h2 style="font-size: 40px; margin-top: 5px; color: #fff; text-align: center">{total_rides:,}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        delay_rate = df_f['delay'].mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <p style="font-size: 20px; color: #EAF6F9; text-align: center">Delay Rate</p>
            <h2 style="font-size: 40px; margin-top: 5px; color: #fff; text-align: center">{delay_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_distance = df_f['Distance'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <p style="font-size: 20px; color: #EAF6F9; text-align: center">Avg Distance per Ride</p>
            <h2 style="font-size: 40px; margin-top: 5px; color: #fff; text-align: center">{avg_distance:.1f} mi</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        busiest_hour = df_f['Pickup_Hour'].value_counts().idxmax()
        st.markdown(f"""
        <div class="metric-card">
            <p style="font-size: 20px; color: #EAF6F9; text-align: center">Busiest Pickup Hour</p>
            <h2 style="font-size: 40px; margin-top: 5px; color: #fff; text-align: center">{busiest_hour}:00</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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

    insights = [
        f"📌 Delay on weekdays is <b>{(wk_delay - we_delay)*100:.1f}%</b> higher than weekends.",
        f"⏰ Busiest pickup hour: <b>{busiest_hour}:00</b>",
        f"📌 Service area with highest delay rate: <b>{top_service}</b> (<b>{service_delay[top_service]*100:.1f}%</b>)",
        f"🏙 City with most delayed rides: <b>{top_city_delay}</b> (<b>{city_delay[top_city_delay]:,}</b> delays)",
        f"⏰ Hour with highest delay rate: <b>{peak_hour}:00</b> (<b>{hourly_delay[peak_hour]*100:.1f}%</b>)",
        f"🏙 Pickup city with most rides: <b>{top_city_rides}</b> (<b>{df_f['PickupCity'].value_counts()[top_city_rides]:,}</b> rides)",
        f"📌 Total delayed rides: <b>{total_delays:,}</b> (<b>{total_delays/len(df_f)*100:.1f}%</b>)",
        f"📌 Delayed rides are on average <b>{avg_dist_delay:.1f} mi</b> vs <b>{avg_dist_all:.1f} mi</b> overall."
    ]

    cols = st.columns(2)
    for i, insight in enumerate(insights):
        with cols[i % 2]:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------
    # Charts in 3 columns
    # -------------------------
    col1_chart, col2_chart, col3_chart = st.columns(3)

# 1️⃣ Doughnut chart — Delays vs No Delays
    delay_counts = df_f['delay_label'].value_counts()
    delay_percent = df_f['delay_label'].value_counts(normalize=True) * 100
    with col1_chart:
        with st.container():
            st.markdown(
                """
                <div style="
                    background-color:#ffffff;
                    padding:15px;
                    border-radius:15px;
                    box-shadow:0 6px 15px rgba(0,0,0,0.1);
                ">
                """, unsafe_allow_html=True
            )
    
            fig_doughnut = go.Figure(data=[go.Pie(
                labels=delay_counts.index,
                values=delay_counts.values,
                hole=0.5,
                hoverinfo="label+percent+value",
                text=[f"{val} ({percent:.1f}%)" for val, percent in zip(delay_counts.values, delay_percent)],
                textinfo="text",
                marker_colors=["#28A745", "#DC3545"]
            )])
            fig_doughnut.update_layout(
                title_text="🚖 Delays vs No Delays (Overall)",
                paper_bgcolor="white",
                plot_bgcolor="white",
                transition={'duration': 1000},
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_doughnut, use_container_width=True)
    
            st.markdown("</div>", unsafe_allow_html=True)
    
    
    # 2️⃣ Bar Chart — % of Delays by Service Area
    service_area_counts = df_f.groupby(['ServiceArea', 'delay_label']).size().reset_index(name='count')
    service_area_totals = service_area_counts.groupby('ServiceArea')['count'].sum().reset_index(name='total')
    service_area_counts = service_area_counts.merge(service_area_totals, on='ServiceArea')
    service_area_counts['percent'] = service_area_counts['count'] / service_area_counts['total'] * 100
    service_area_counts['label'] = service_area_counts.apply(lambda x: f"{x['count']} ({x['percent']:.1f}%)", axis=1)
    
    with col2_chart:
        with st.container():
            st.markdown(
                """
                <div style="
                    background-color:#ffffff;
                    padding:15px;
                    border-radius:15px;
                    box-shadow:0 6px 15px rgba(0,0,0,0.1);
                ">
                """, unsafe_allow_html=True
            )
    
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
    
            st.markdown("</div>", unsafe_allow_html=True)
    
    
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
            st.markdown(
                """
                <div style="
                    background-color:#ffffff;
                    padding:15px;
                    border-radius:15px;
                    box-shadow:0 6px 15px rgba(0,0,0,0.1);
                ">
                """, unsafe_allow_html=True
            )
    
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
    
            st.markdown("</div>", unsafe_allow_html=True)
    
    
    # 4️⃣ Full-width — Delays by Hour
    st.markdown('<div style="margin-top:20px;">', unsafe_allow_html=True)  # Add spacing before full-width card
    with st.container():
        st.markdown(
            """
            <div style="
                background-color:#ffffff;
                padding:15px;
                border-radius:15px;
                box-shadow:0 6px 15px rgba(0,0,0,0.1);
            ">
            """, unsafe_allow_html=True
        )
    
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
        st.markdown("</div>", unsafe_allow_html=True)



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

        st.markdown('<div class="big-card">', unsafe_allow_html=True)
        fig1 = px.bar(
            delay_counts,
            x='PickupDelay_Flag',
            y='count',
            text='label',
            color='PickupDelay_Flag',
            color_discrete_sequence=['#0069AC'],
            title="🚖 Pickup Delay Flags (%)"
        )
        fig1.update_traces(textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # ==========================
    # 2. Service Acknowledgement
    # ==========================
    with c2:
        ack_counts = kpi['ServiceAck_Flag'].value_counts().reset_index()
        ack_counts.columns = ['ServiceAck_Flag', 'count']
        ack_counts['percent'] = (ack_counts['count'] / total_trips * 100).round(1)
        ack_counts['label'] = ack_counts.apply(lambda x: f"{x['count']} ({x['percent']}%)", axis=1)

        st.markdown('<div class="big-card">', unsafe_allow_html=True)
        fig2 = px.bar(
            ack_counts,
            x='ServiceAck_Flag',
            y='count',
            text='label',
            color='ServiceAck_Flag',
            color_discrete_sequence=['#0069AC'],
            title="📝 Service Acknowledgement Issue (%)"
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # ==========================
    # 3. Alert Levels
    # ==========================
    with c3:
        alert_counts = kpi['AlertLevel'].value_counts().reset_index()
        alert_counts.columns = ['AlertLevel', 'count']
        alert_counts['percent'] = (alert_counts['count'] / total_trips * 100).round(1)
        alert_counts['label'] = alert_counts.apply(lambda x: f"{x['count']} ({x['percent']}%)", axis=1)

        st.markdown('<div class="big-card">', unsafe_allow_html=True)
        fig3 = px.bar(
            alert_counts,
            x='AlertLevel',
            y='count',
            text='label',
            color='AlertLevel',
            title="🚨 Alerts (%)",
            color_discrete_sequence=["#31C035", "#EC2323", "#0069AC"]
        )
        fig3.update_traces(textposition='outside')
        fig3.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # ============================================
    # ROW 2 – HEATMAP + STACKED BAR
    # ============================================
    d1, d2, d3 = st.columns([1,1,1])

    # ==========================
    # 4. Discrete Heatmap
        # ==========================
    with d1:
        # 1. Extract time features
        kpi['Pickup_Hour'] = pd.to_datetime(kpi['Pickup DateTime']).dt.hour
        kpi['Weekday'] = pd.to_datetime(kpi['Pickup DateTime']).dt.day_name()
    
        # 2. Breach indicator
        kpi['IsRed'] = kpi['AlertLevel'].str.startswith("RED")
        heat = kpi.groupby(['Weekday', 'Pickup_Hour'])['IsRed'].mean().reset_index()
        heat['Pct'] = heat['IsRed'] * 100


        st.markdown('<div class="big-card">', unsafe_allow_html=True)
        colors = ['#d4f4dd', '#a8e6a3', '#7cd66a', '#ffb366', '#ff704d']
        fig4 = px.density_heatmap(
        heat,
        x='Pickup_Hour',
        y='Weekday',
        z='Pct',
        color_continuous_scale=colors,
        title="🔥 Heatmap: Red Alert Breach Rate by Hour & Weekday")
        fig4.update_layout(
        title="🔥 Heatmap: Red Alerts by Hour & Weekday",
        xaxis_title="Pickup Hour",
        yaxis_title="Day of Week (0=Mon)",
        coloraxis_colorbar_title="% Red Alerts" )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        
    # ==========================
    # 5. Stacked Bar (Service Area)
    # ==========================
    with d2:
        alert_area = kpi.groupby(['ServiceAreaCode', 'AlertLevel']).size().reset_index(name='count')
        alert_area['percent'] = alert_area.groupby('ServiceAreaCode')['count'].transform(lambda x: x / x.sum() * 100)       
        alert_area['label'] = alert_area.apply(lambda x: f"{x['count']} ({x['percent']:.1f}%)", axis=1)

        st.markdown('<div class="big-card">', unsafe_allow_html=True)
        fig5 = px.bar(
            alert_area,
            x='ServiceAreaCode',
            y='count',
            color='AlertLevel',
            text='label',
            barmode='stack',
            title="📦 Alert Breakdown by Service Area (%)",
            color_discrete_sequence=["#31C035", "#0069AC", "#EC2323"]
        )
        fig5.update_traces(textposition='inside')
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)



# =========================
# TAB 3 — GEO
# =========================
with tab_geo:
    st.subheader("🗺 Pickup City Geo Map (Delay vs No Delay)")

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
        return city_coords.get(city, (40.71 + random.uniform(-0.3,0.3), -74 + random.uniform(-0.3,0.3)))

    df_map = df_f.copy()
    df_map["lat"] = df_map["PickupCity"].apply(lambda x: coord(x)[0])
    df_map["lon"] = df_map["PickupCity"].apply(lambda x: coord(x)[1])

    fig = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        color="delay_label",
        hover_name="PickupCity",
        zoom=4,
        height=600,
        color_discrete_map={"No Delay": "#28A745", "Delay": "#DC3545"},
    )

    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)