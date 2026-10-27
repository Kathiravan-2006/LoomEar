"""
LoomEar: State-of-the-Art Acoustic Predictive Maintenance & 3D Digital Twin Console.
Features live acoustic monitoring, pick-cycle residual explainability, multi-day trend analysis,
plant fleet matrix, dual currency (INR ₹ / USD $) ROI simulator, and an interactive 3D Power Loom Digital Twin.
"""

from io import BytesIO
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components

# LoomEar imports
from src.loomear.audio.cycles import (
    build_healthy_template,
    compute_cycle_residual,
    compute_log_mel_spectrogram,
    segment_cycles,
)
from src.loomear.audio.preprocess import clean_audio
from src.loomear.data.simulator import SyntheticLoomSimulator
from src.loomear.eval.explain import generate_residual_explanation
from src.loomear.health.cost_model import MaintenanceCostModel
from src.loomear.health.decision import MaintenanceDecisionEngine, LoomState
from src.loomear.health.index import compute_health_index, smooth_health_index

# Page Configuration
st.set_page_config(
    page_title="LoomEar | 3D Acoustic Predictive Maintenance",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced Modern Cyber-Industrial Theme & Glassmorphic CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    :root {
        --bg-dark: #0B1120;
        --card-bg: rgba(17, 24, 39, 0.85);
        --accent-cyan: #06B6D4;
        --accent-emerald: #10B981;
        --accent-amber: #F59E0B;
        --accent-rose: #F43F5E;
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0F172A 0%, #020617 90%);
        color: #F8FAFC;
    }
    
    /* Header Gradient */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    
    .hero-sub {
        color: #94A3B8;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Glassmorphic Metric Cards */
    .glow-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 18px 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glow-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    
    /* Glowing Status Badges */
    .badge-healthy {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
    }
    
    .badge-watch {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.25);
    }
    
    .badge-act {
        background: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.45);
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.35);
        animation: pulse-glow 2s infinite;
    }
    
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); }
        50% { box-shadow: 0 0 25px rgba(239, 68, 68, 0.6); }
    }
    
    .damage-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 12px 0;">
        <h2 style="margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🧵 LoomEar</h2>
        <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.85rem; font-weight: 600;">AI ACOUSTIC DIGITAL TWIN</p>
    </div>
    """,
    unsafe_allow_html=True,
)

nav = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Live Acoustic Monitor",
        "2. 3D Power Loom Digital Twin",
        "3. Cycle & Residual Explainability",
        "4. Health Trend & Lead-Time",
        "5. Fleet Overview & Plant Matrix",
        "6. Financial ROI Simulator (₹ INR / $ USD)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Machinery Setup")
selected_machine = st.sidebar.selectbox(
    "Monitored Loom",
    [
        "LOOM-01 (Rapier)",
        "LOOM-02 (Rapier)",
        "LOOM-03 (Projectile)",
        "LOOM-04 (Air-Jet)",
        "LOOM-05 (Rapier)",
        "LOOM-06 (Rapier)",
        "LOOM-07 (Air-Jet)",
        "LOOM-08 (Projectile)",
    ],
)
machine_id = selected_machine.split(" ")[0]

# Simulator Singleton
@st.cache_resource
def get_simulator():
    return SyntheticLoomSimulator(sample_rate=16000, random_seed=42)

sim = get_simulator()


# ----------------------------------------------------
# TAB 1: LIVE ACOUSTIC MONITOR
# ----------------------------------------------------
if nav == "1. Live Acoustic Monitor":
    st.markdown(
        f"""
        <div class="hero-banner">
            <h1 class="hero-title">🎧 Real-Time Acoustic Condition Monitor</h1>
            <p class="hero-sub">Continuously capturing airborne sound on <b>{selected_machine}</b> with pick-cycle synchronization and neural anomaly detection.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 🎙️ Audio Telemetry Feed")
        source_mode = st.radio("Signal Stream Source:", ["Simulate Live Loom Sensor", "Upload WAV Recording"])
        
        if source_mode == "Simulate Live Loom Sensor":
            sim_condition = st.selectbox("Machine Operational Health Mode:", [
                "🟢 Normal Operational (Healthy)",
                "🟡 Developing Bearing Wear (Watch)",
                "🔴 Critical Bearing Seizure (Act)",
                "🟡 Timing Drift / Phase Lag (Watch)",
                "🟡 Belt Squeal & Slip (Watch)",
                "🔴 Beat-up Reed Looseness (Act)",
            ])
            
            # Map selection to simulator parameters
            if "Normal" in sim_condition:
                audio, _ = sim.generate_healthy_loom(duration_sec=4.0, rpm=212.0)
                audio = sim.mix_ambience_snr(audio, snr_db=18.0)
            elif "Developing Bearing" in sim_condition:
                c, _ = sim.generate_healthy_loom(duration_sec=4.0, rpm=212.0)
                f = sim.inject_fault(c, "bearing_wear", severity=0.40, rpm=212.0)
                audio = sim.mix_ambience_snr(f, snr_db=15.0)
            elif "Critical Bearing" in sim_condition:
                c, _ = sim.generate_healthy_loom(duration_sec=4.0, rpm=212.0)
                f = sim.inject_fault(c, "bearing_wear", severity=0.88, rpm=212.0)
                audio = sim.mix_ambience_snr(f, snr_db=10.0)
            elif "Timing" in sim_condition:
                c, _ = sim.generate_healthy_loom(duration_sec=4.0, rpm=212.0)
                f = sim.inject_fault(c, "timing_drift", severity=0.65, rpm=212.0)
                audio = sim.mix_ambience_snr(f, snr_db=15.0)
            elif "Belt" in sim_condition:
                c, _ = sim.generate_healthy_loom(duration_sec=4.0, rpm=212.0)
                f = sim.inject_fault(c, "belt_squeal", severity=0.60, rpm=212.0)
                audio = sim.mix_ambience_snr(f, snr_db=15.0)
            else:
                c, _ = sim.generate_healthy_loom(duration_sec=4.0, rpm=212.0)
                f = sim.inject_fault(c, "beatup_looseness", severity=0.82, rpm=212.0)
                audio = sim.mix_ambience_snr(f, snr_db=12.0)
            sr = 16000
        else:
            uploaded_file = st.file_uploader("Upload Acoustic WAV (.wav)", type=["wav", "wave"])
            if uploaded_file is not None:
                audio, sr = sf.read(uploaded_file, dtype="float32")
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
            else:
                st.info("Awaiting audio file upload. Defaulting to healthy live stream.")
                audio, _ = sim.generate_healthy_loom(duration_sec=4.0, rpm=212.0)
                sr = 16000

        # Audio Processing
        cleaned = clean_audio(audio, sample_rate=sr)
        cycles_res = segment_cycles(cleaned, sample_rate=sr)
        template = build_healthy_template(cycles_res.normalized_cycles, sample_rate=sr)
        res_map, anomaly_score = compute_cycle_residual(cycles_res.normalized_cycles[0], template, sample_rate=sr)

        # 0-100 Health Index
        h_baseline = np.array([0.05, 0.08, 0.12, 0.15, 0.18, 0.22, 0.25])
        h_index = float(compute_health_index(np.array([anomaly_score]), h_baseline)[0])

        if h_index < 60.0:
            badge_html = '<span class="badge-healthy">🟢 HEALTHY (Normal)</span>'
        elif h_index < 80.0:
            badge_html = '<span class="badge-watch">🟡 WATCH (Incipient Wear Detected)</span>'
        else:
            badge_html = '<span class="badge-act">🔴 ACT (Critical Breakdown Hazard)</span>'

        st.markdown(f"#### Machine Status: {badge_html}", unsafe_allow_html=True)
        st.markdown(f"**Health Index Score:** `{h_index:.1f} / 100`")

        # Audio Player
        buf = BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        st.audio(buf.getvalue(), format="audio/wav")

    with col2:
        st.markdown("### 📊 Real-Time Kinematic Telemetry")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Loom Speed", f"{cycles_res.estimated_rpm:.1f} PPM", delta="Normal", delta_color="normal")
        m2.metric("Pick Period", f"{cycles_res.estimated_period_sec*1000:.1f} ms")
        m3.metric("Cycle Sync Conf.", f"{cycles_res.confidence*100:.1f}%")
        m4.metric("Cycles Analyzed", f"{len(cycles_res.cycles)}")

        # Vibrant Health Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=h_index,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Acoustic Degradation Index", 'font': {'size': 18, 'color': '#F8FAFC'}},
            number={'font': {'size': 42, 'color': '#38BDF8'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#64748B"},
                'bar': {'color': "#38BDF8", 'thickness': 0.3},
                'bgcolor': "rgba(15, 23, 42, 0.8)",
                'borderwidth': 2,
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'steps': [
                    {'range': [0, 60], 'color': "rgba(16, 185, 129, 0.3)"},
                    {'range': [60, 80], 'color': "rgba(245, 158, 11, 0.35)"},
                    {'range': [80, 100], 'color': "rgba(239, 68, 68, 0.45)"},
                ],
                'threshold': {
                    'line': {'color': "#EF4444", 'width': 5},
                    'thickness': 0.85,
                    'value': 80.0,
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=240,
            margin=dict(l=20, r=20, t=40, b=10),
            font=dict(color="#F8FAFC"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Waveform with cycle boundaries
    st.markdown("### 🌊 Pick-Cycle-Synchronous Waveform & Impact Anchors")
    t_axis = np.arange(len(cleaned)) / sr
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(
        x=t_axis, y=cleaned, mode="lines", name="Acoustic Signal",
        line=dict(color="#06B6D4", width=1.2),
    ))
    
    for b_idx in cycles_res.cycle_boundaries[:12]:
        b_time = b_idx / sr
        fig_wave.add_vline(x=b_time, line_width=1.5, line_dash="dot", line_color="#F43F5E")

    fig_wave.update_layout(
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#94A3B8"),
        title="Waveform with Pick-Cycle Beat-up Impact Anchors (Red Dotted Lines)",
        xaxis=dict(title="Time (seconds)", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Normalized Amplitude", gridcolor="rgba(255,255,255,0.06)"),
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_wave, use_container_width=True)


# ----------------------------------------------------
# TAB 2: 3D POWER LOOM DIGITAL TWIN & DAMAGE VIEWER
# ----------------------------------------------------
elif nav == "2. 3D Power Loom Digital Twin":
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">🕹️ 3D Power Loom Digital Twin & Damage Inspector</h1>
            <p class="hero-sub">Interactive spatial visualization of mechanical sub-assemblies, acoustic stress heatmaps, component-level damage localization, and real-time loss tracking.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_col1, t_col2 = st.columns([2, 1])

    with t_col2:
        st.markdown("### ⚙️ Component Inspector")
        target_component = st.selectbox(
            "Inspect Sub-Assembly:",
            [
                "1. Main Crankshaft & Roller Bearings",
                "2. Sley Sword & Reed Beat-Up Mechanism",
                "3. Rapier Head & Weft Insertion Guide",
                "4. Drive Motor & Belt Pulley Tensioner",
                "5. Shedding Cams & Heald Wire Harness",
            ],
        )

        wear_severity = st.slider("Simulate Wear Severity (%)", min_value=0, max_value=100, value=75, step=5)
        
        # Damage Metrics & Loss per Component
        if "Crankshaft" in target_component:
            comp_name = "Main Crankshaft Bearing"
            loss_rate_inr = wear_severity * 320
            loss_rate_usd = wear_severity * 3.8
            vib_amp = 0.4 + (wear_severity / 100.0) * 4.2
            temp_c = 42.0 + (wear_severity / 100.0) * 46.0
            defect_desc = "Spalling on inner bearing race with ball-pass frequency acoustic surging (3.5 - 7.5 kHz)."
            color_theme = "#EF4444" if wear_severity > 60 else ("#F59E0B" if wear_severity > 30 else "#10B981")
        elif "Sley" in target_component:
            comp_name = "Sley & Reed Beat-Up"
            loss_rate_inr = wear_severity * 280
            loss_rate_usd = wear_severity * 3.3
            vib_amp = 0.8 + (wear_severity / 100.0) * 5.6
            temp_c = 36.0 + (wear_severity / 100.0) * 28.0
            defect_desc = "Mechanical play in sley sword linkage causing doubled impact thuds and uneven weft compaction."
            color_theme = "#EF4444" if wear_severity > 60 else ("#F59E0B" if wear_severity > 30 else "#10B981")
        elif "Rapier" in target_component:
            comp_name = "Rapier Weft Guide"
            loss_rate_inr = wear_severity * 240
            loss_rate_usd = wear_severity * 2.9
            vib_amp = 0.3 + (wear_severity / 100.0) * 3.8
            temp_c = 38.0 + (wear_severity / 100.0) * 35.0
            defect_desc = "Friction scuffing along rapier guide tape increasing yarn fraying and stoppage probability."
            color_theme = "#EF4444" if wear_severity > 60 else ("#F59E0B" if wear_severity > 30 else "#10B981")
        elif "Drive" in target_component:
            comp_name = "Belt Pulley Drive"
            loss_rate_inr = wear_severity * 190
            loss_rate_usd = wear_severity * 2.3
            vib_amp = 0.5 + (wear_severity / 100.0) * 4.5
            temp_c = 45.0 + (wear_severity / 100.0) * 52.0
            defect_desc = "Glazed V-belt slipping under peak shedding torque with high-pitch tonal squeal."
            color_theme = "#EF4444" if wear_severity > 60 else ("#F59E0B" if wear_severity > 30 else "#10B981")
        else:
            comp_name = "Shedding Harness"
            loss_rate_inr = wear_severity * 160
            loss_rate_usd = wear_severity * 1.9
            vib_amp = 0.2 + (wear_severity / 100.0) * 3.1
            temp_c = 32.0 + (wear_severity / 100.0) * 20.0
            defect_desc = "Heald wire micro-fatigue and timing phase shift leading to warp thread snags."
            color_theme = "#EF4444" if wear_severity > 60 else ("#F59E0B" if wear_severity > 30 else "#10B981")

        st.markdown(
            f"""
            <div class="damage-box" style="border-color: {color_theme};">
                <h4 style="color: {color_theme}; margin: 0 0 8px 0;">⚠️ {comp_name} Damage Assessment</h4>
                <p style="color: #CBD5E1; font-size: 0.92rem; margin-bottom: 12px;">{defect_desc}</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.9rem;">
                    <div><b>Acoustic Stress:</b> <span style="color: {color_theme};">{wear_severity}%</span></div>
                    <div><b>Operating Temp:</b> <span>{temp_c:.1f} °C</span></div>
                    <div><b>Vibration Level:</b> <span>{vib_amp:.2f} mm/s</span></div>
                    <div><b>Loss Accrual:</b> <span style="color: #F87171;">₹{loss_rate_inr:,.0f}/hr (${loss_rate_usd:.1f}/hr)</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t_col1:
        st.markdown("### 🌐 3D Interactive Power Loom Spatial Model")
        
        # Build interactive Plotly 3D Digital Twin Model
        # Loom Structural Frame (Cast iron chassis)
        fig_3d = go.Figure()

        # 1. Main Loom Frame Box / Pillars
        frame_x = [0, 4, 4, 0, 0, 0, 4, 4, 0, 0, 4, 4, 4, 4, 0, 0]
        frame_y = [0, 0, 3, 3, 0, 0, 0, 3, 3, 0, 0, 0, 3, 3, 3, 3]
        frame_z = [0, 0, 0, 0, 0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 0, 0, 2.5, 2.5, 0]
        fig_3d.add_trace(go.Scatter3d(
            x=frame_x, y=frame_y, z=frame_z,
            mode='lines',
            line=dict(color='#475569', width=6),
            name='Heavy Cast-Iron Chassis',
            hoverinfo='name'
        ))

        # 2. Main Crankshaft & Bearings (Cylinder along X axis at Y=1.5, Z=1.8)
        crank_x = np.linspace(0.2, 3.8, 40)
        crank_y = np.full_like(crank_x, 1.5)
        crank_z = np.full_like(crank_x, 1.8)
        
        bearing_heat = wear_severity if "Crankshaft" in target_component else 20.0
        b_color = '#EF4444' if bearing_heat > 60 else ('#F59E0B' if bearing_heat > 30 else '#10B981')
        
        fig_3d.add_trace(go.Scatter3d(
            x=crank_x, y=crank_y, z=crank_z,
            mode='lines+markers',
            line=dict(color=b_color, width=12),
            marker=dict(size=4, color=b_color),
            name='Main Crankshaft (Drive Shaft)',
        ))

        # Bearing Blocks (Spheres at ends)
        fig_3d.add_trace(go.Scatter3d(
            x=[0.4, 3.6], y=[1.5, 1.5], z=[1.8, 1.8],
            mode='markers',
            marker=dict(size=14, color=b_color, symbol='diamond'),
            name='Roller Bearing Housing (Acoustic Hotspot)',
        ))

        # 3. Sley Sword & Reed Beat-Up Mechanism (Reciprocating rocker at Y=1.8 to 2.2, Z=1.0)
        sley_heat = wear_severity if "Sley" in target_component else 25.0
        s_color = '#EF4444' if sley_heat > 60 else ('#F59E0B' if sley_heat > 30 else '#38BDF8')
        sley_x = np.linspace(0.5, 3.5, 30)
        sley_y = np.full_like(sley_x, 2.0)
        sley_z = np.full_like(sley_x, 1.2)
        fig_3d.add_trace(go.Scatter3d(
            x=sley_x, y=sley_y, z=sley_z,
            mode='lines',
            line=dict(color=s_color, width=14),
            name='Sley Sword & Beat-up Reed',
        ))

        # 4. Warp Yarn Sheet (Planar grid flowing from back Z=1.0, Y=0.2 to front Z=1.2, Y=2.8)
        yy, xx = np.meshgrid(np.linspace(0.4, 2.6, 6), np.linspace(0.6, 3.4, 8))
        zz = 1.0 + 0.1 * yy
        fig_3d.add_trace(go.Surface(
            x=xx, y=yy, z=zz,
            colorscale=[[0, 'rgba(99, 102, 241, 0.4)'], [1, 'rgba(168, 85, 247, 0.6)']],
            showscale=False,
            name='Warp Yarn Sheet (Fabric Path)',
        ))

        # 5. Microphone Acoustic Sensor Probe
        fig_3d.add_trace(go.Scatter3d(
            x=[2.0], y=[2.6], z=[1.6],
            mode='markers+text',
            marker=dict(size=10, color='#06B6D4', symbol='circle'),
            text=['🎤 LoomEar Mic Sensor (35cm)'],
            textposition='top center',
            textfont=dict(color='#38BDF8', size=12),
            name='Acoustic Sensor Probe',
        ))

        # Acoustic Wave Expansion Rings from Fault Hotspot
        if wear_severity > 30:
            ring_theta = np.linspace(0, 2*np.pi, 30)
            ring_r = 0.5 + 0.3 * (wear_severity / 100.0)
            ring_x = 0.4 + ring_r * np.cos(ring_theta)
            ring_z = 1.8 + ring_r * np.sin(ring_theta)
            ring_y = np.full_like(ring_x, 1.5)
            fig_3d.add_trace(go.Scatter3d(
                x=ring_x, y=ring_y, z=ring_z,
                mode='lines',
                line=dict(color='#F87171', width=4, dash='dash'),
                name='Acoustic Shockwave Radiation',
            ))

        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title='Width (m)', backgroundcolor="rgba(15, 23, 42, 0.8)", gridcolor="#334155", showbackground=True),
                yaxis=dict(title='Depth (m)', backgroundcolor="rgba(15, 23, 42, 0.8)", gridcolor="#334155", showbackground=True),
                zaxis=dict(title='Height (m)', backgroundcolor="rgba(15, 23, 42, 0.8)", gridcolor="#334155", showbackground=True),
                camera=dict(eye=dict(x=1.6, y=-1.8, z=1.2)),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=460,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(x=0, y=1, bgcolor="rgba(15, 23, 42, 0.7)", font=dict(color="#CBD5E1")),
        )
        st.plotly_chart(fig_3d, use_container_width=True)


# ----------------------------------------------------
# TAB 3: EXPLAINABILITY & RESIDUAL HEATMAP
# ----------------------------------------------------
elif nav == "3. Cycle & Residual Explainability":
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">🔬 Pick-Cycle Residual Heatmap & Mechanic Explainability</h1>
            <p class="hero-sub">Standardizing live cycles against the learned healthy baseline template isolates exactly <b>where in the pick</b> and <b>at which frequencies</b> degradation occurs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c_healthy, _ = sim.generate_healthy_loom(duration_sec=3.0, rpm=210.0)
    c_fault = sim.inject_fault(c_healthy, "bearing_wear", severity=0.75, rpm=210.0)
    
    res_h = segment_cycles(clean_audio(c_healthy), sample_rate=16000)
    res_f = segment_cycles(clean_audio(c_fault), sample_rate=16000)
    
    template = build_healthy_template(res_h.normalized_cycles, sample_rate=16000)
    residual_map, score = compute_cycle_residual(res_f.normalized_cycles[0], template, sample_rate=16000)
    explanation = generate_residual_explanation(residual_map)

    with c1:
        st.markdown("### 🟢 Healthy Baseline Pick Template")
        fig_t = px.imshow(
            template["mean_spec"],
            labels=dict(x="Normalized Cycle Phase (%)", y="Mel Frequency Bin", color="Log Power"),
            color_continuous_scale="Viridis",
            origin="lower",
        )
        fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"), height=300)
        st.plotly_chart(fig_t, use_container_width=True)

    with c2:
        st.markdown("### 🔴 Standardized Residual Error Map")
        fig_r = px.imshow(
            residual_map,
            labels=dict(x="Normalized Cycle Phase (%)", y="Mel Frequency Bin", color="Z-Score"),
            color_continuous_scale="RdBu_r",
            origin="lower",
        )
        fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"), height=300)
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown(
        f"""
        <div class="damage-box" style="border-color: rgba(56, 189, 248, 0.4);">
            <h4 style="color: #38BDF8; margin: 0 0 6px 0;">📋 Automated Mechanic Diagnostic Explanation</h4>
            <p style="color: #F1F5F9; font-size: 1.05rem; margin: 0;"><b>Diagnosis:</b> {explanation['plain_english_summary']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------
# TAB 4: HEALTH TREND & LEAD TIME
# ----------------------------------------------------
elif nav == "4. Health Trend & Lead-Time":
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">📈 14-Day Health Degradation Trajectory & Lead Time</h1>
            <p class="hero-sub">Proving predictive maintenance capability: tracks the rising degradation curve to deliver actionable lead time before breakdown.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    days = np.linspace(1, 14, 140)
    raw_trend = 22.0 + 3.0 * np.random.randn(len(days)) + 55.0 / (1.0 + np.exp(-1.1 * (days - 10.5)))
    smoothed_trend = smooth_health_index(raw_trend, alpha=0.20)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=days, y=raw_trend, mode="markers", name="Raw Cycle Score",
        marker=dict(color="#64748B", size=4, opacity=0.6)
    ))
    fig_trend.add_trace(go.Scatter(
        x=days, y=smoothed_trend, mode="lines", name="EWMA Health Index",
        line=dict(color="#38BDF8", width=3.5)
    ))

    fig_trend.add_hline(y=60.0, line_dash="dash", line_color="#F59E0B", annotation_text="WATCH Threshold (60)", annotation_position="top left", annotation_font_color="#F59E0B")
    fig_trend.add_hline(y=80.0, line_dash="dash", line_color="#EF4444", annotation_text="ACT Threshold (80)", annotation_position="top left", annotation_font_color="#EF4444")

    watch_day = days[np.where(smoothed_trend >= 60.0)[0][0]]
    failure_day = 14.0
    lead_time_days = failure_day - watch_day

    fig_trend.add_vrect(
        x0=watch_day, x1=failure_day, fillcolor="rgba(245, 158, 11, 0.15)",
        line_width=0, annotation_text=f"Early Warning Window ({lead_time_days:.1f} Days / {lead_time_days*24:.0f} Hrs Lead Time)",
        annotation_position="top right", annotation_font_color="#FBBF24"
    )

    fig_trend.update_layout(
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#CBD5E1"),
        xaxis=dict(title="Operation Timeline (Production Days)", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Loom Health Index (0 - 100)", range=[0, 100], gridcolor="rgba(255,255,255,0.06)"),
        height=450,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.success(f"✅ **Early Warning Verified:** System triggered `WATCH` status **{lead_time_days*24:.0f} hours before failure**, allowing off-shift bearing lubrication without unplanned downtime.")


# ----------------------------------------------------
# TAB 5: FLEET OVERVIEW & PLANT MATRIX (FIXED & ENHANCED)
# ----------------------------------------------------
elif nav == "5. Fleet Overview & Plant Matrix":
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">🏭 Plant Fleet Health Matrix & Risk Ranking</h1>
            <p class="hero-sub">Live condition monitoring across all production weaving looms in Weaving Shed #2 (Tirupur / Surat Mill Cluster).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fleet_data = [
        {"Machine ID": "LOOM-01", "Type": "Rapier", "Speed (PPM)": 210, "Shift": "Shift A", "Health Index": 24.5, "Status": "Healthy", "Dominant Acoustic Anomaly": "None (Clean Harmonics)", "Estimated Risk": "Low (0.05)", "Action": "Normal Operation"},
        {"Machine ID": "LOOM-02", "Type": "Rapier", "Speed (PPM)": 215, "Shift": "Shift A", "Health Index": 32.1, "Status": "Healthy", "Dominant Acoustic Anomaly": "None (Normal Baseline)", "Estimated Risk": "Low (0.08)", "Action": "Normal Operation"},
        {"Machine ID": "LOOM-03", "Type": "Projectile", "Speed (PPM)": 195, "Shift": "Shift A", "Health Index": 68.4, "Status": "Watch", "Dominant Acoustic Anomaly": "V-Belt Slip & High-Band Squeal", "Estimated Risk": "Moderate (0.64)", "Action": "Check belt tension on break"},
        {"Machine ID": "LOOM-04", "Type": "Air-Jet", "Speed (PPM)": 240, "Shift": "Shift A", "Health Index": 84.7, "Status": "Act", "Dominant Acoustic Anomaly": "Main Bearing Dry Friction & Clicks", "Estimated Risk": "Critical (0.89)", "Action": "Schedule immediate lubrication"},
        {"Machine ID": "LOOM-05", "Type": "Rapier", "Speed (PPM)": 210, "Shift": "Shift B", "Health Index": 28.0, "Status": "Healthy", "Dominant Acoustic Anomaly": "None (Normal Baseline)", "Estimated Risk": "Low (0.06)", "Action": "Normal Operation"},
        {"Machine ID": "LOOM-06", "Type": "Rapier", "Speed (PPM)": 208, "Shift": "Shift B", "Health Index": 41.2, "Status": "Healthy", "Dominant Acoustic Anomaly": "Minor Shedding Noise", "Estimated Risk": "Low (0.15)", "Action": "Normal Operation"},
        {"Machine ID": "LOOM-07", "Type": "Air-Jet", "Speed (PPM)": 245, "Shift": "Shift C", "Health Index": 72.8, "Status": "Watch", "Dominant Acoustic Anomaly": "Sley Reed Looseness", "Estimated Risk": "Moderate (0.71)", "Action": "Tighten reed bolts during maintenance"},
        {"Machine ID": "LOOM-08", "Type": "Projectile", "Speed (PPM)": 190, "Shift": "Shift C", "Health Index": 21.0, "Status": "Healthy", "Dominant Acoustic Anomaly": "None (Clean Harmonics)", "Estimated Risk": "Low (0.04)", "Action": "Normal Operation"},
    ]
    df_fleet = pd.DataFrame(fleet_data)

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total Monitored Looms", "8 Looms")
    f2.metric("🟢 Healthy State", "5 Looms", "62.5%")
    f3.metric("🟡 Watch Warning", "2 Looms", "25.0%", delta_color="off")
    f4.metric("🔴 Critical Act Required", "1 Loom", "12.5%", delta_color="inverse")

    # Filter selector
    filter_status = st.multiselect("Filter by Operational State:", ["Healthy", "Watch", "Act"], default=["Healthy", "Watch", "Act"])
    filtered_df = df_fleet[df_fleet["Status"].isin(filter_status)]

    st.markdown("### 📋 Real-Time Machine Health Registry")
    
    # Modern Native Streamlit Interactive Data Table with Progress Gauges & Custom Column Configs
    df_display = filtered_df[[
        "Machine ID", "Type", "Speed (PPM)", "Shift", "Health Index", "Status", "Dominant Acoustic Anomaly", "Action"
    ]].copy()

    # Map status to emoji icons
    df_display["Status"] = df_display["Status"].map({
        "Healthy": "🟢 HEALTHY",
        "Watch": "🟡 WATCH (Warning)",
        "Act": "🔴 ACT (Critical)",
    }).fillna(df_display["Status"])

    st.dataframe(
        df_display,
        column_config={
            "Health Index": st.column_config.ProgressColumn(
                "Degradation Index",
                help="0 to 60: Healthy | 60 to 80: Watch | 80 to 100: Critical Act",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
            "Status": st.column_config.TextColumn(
                "Operational State",
                width="medium",
            ),
            "Speed (PPM)": st.column_config.NumberColumn(
                "Speed (PPM)",
                format="%d PPM",
            ),
            "Action": st.column_config.TextColumn(
                "Maintenance Action",
                width="large",
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

    # Interactive Machine Drill-down Cards
    st.markdown("### 🔍 Filtered Machine Status Cards")
    card_cols = st.columns(min(4, max(1, len(filtered_df))))
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        col = card_cols[i % len(card_cols)]
        status_color = "#34D399" if row["Status"] == "Healthy" else ("#FBBF24" if row["Status"] == "Watch" else "#F87171")
        badge_cls = "badge-healthy" if row["Status"] == "Healthy" else ("badge-watch" if row["Status"] == "Watch" else "badge-act")
        
        with col:
            st.markdown(
                f"""
                <div class="glow-card" style="border-left: 4px solid {status_color}; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 800; font-size: 1.1rem; color: #F8FAFC;">{row["Machine ID"]}</span>
                        <span class="{badge_cls}">{row["Status"].upper()}</span>
                    </div>
                    <div style="font-size: 0.88rem; color: #94A3B8;"><b>Type:</b> {row["Type"]} ({row["Speed (PPM)"]} PPM)</div>
                    <div style="font-size: 0.88rem; color: #94A3B8;"><b>Shift:</b> {row["Shift"]}</div>
                    <div style="font-size: 0.92rem; color: {status_color}; margin-top: 6px;"><b>Score:</b> {row["Health Index"]:.1f}/100</div>
                    <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px;">{row["Dominant Acoustic Anomaly"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Risk Distribution Bar Chart
    st.markdown("### 📊 Fleet Degradation Index Breakdown")
    fig_fleet = px.bar(
        df_fleet, x="Machine ID", y="Health Index", color="Status",
        color_discrete_map={"Healthy": "#10B981", "Watch": "#F59E0B", "Act": "#EF4444"},
        text_auto=".1f",
        title="Fleet Machine Risk Index Comparison",
    )
    fig_fleet.update_layout(
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#CBD5E1"),
        height=320,
    )
    st.plotly_chart(fig_fleet, use_container_width=True)


# ----------------------------------------------------
# TAB 6: FINANCIAL ROI SIMULATOR (₹ INR / $ USD)
# ----------------------------------------------------
elif nav == "6. Financial ROI Simulator (₹ INR / $ USD)":
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">💰 Maintenance Economics & ROI Simulator</h1>
            <p class="hero-sub">Calculate quantifiable annual cost savings and return on investment in <b>Indian Rupees (₹ INR)</b> and <b>US Dollars ($ USD)</b> for textile weaving mills.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    currency_col, _ = st.columns([1, 2])
    with currency_col:
        currency = st.radio("Select Currency:", ["🇮🇳 Indian Rupee (₹ INR)", "🇺🇸 US Dollar ($ USD)"], horizontal=True)
    is_inr = "INR" in currency

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🎛️ Mill Cost Parameters")
        num_looms = st.slider("Total Power Looms in Shed", min_value=4, max_value=120, value=24, step=2)

        if is_inr:
            downtime_cost = st.slider("Unplanned Downtime Cost (₹ / hour)", min_value=2000.0, max_value=30000.0, value=10000.0, step=500.0)
            sensor_cost = st.slider("Annual Sensor Hardware Cost (₹ / loom)", min_value=1000.0, max_value=12000.0, value=5200.0, step=200.0)
            fa_cost = st.slider("False Alarm Labor Cost (₹ / event)", min_value=200.0, max_value=4000.0, value=2000.0, step=100.0)
            curr_sym = "₹"
        else:
            downtime_cost = st.slider("Unplanned Downtime Cost ($ / hour)", min_value=30.0, max_value=300.0, value=120.0, step=5.0)
            sensor_cost = st.slider("Annual Sensor Hardware Cost ($ / loom)", min_value=20.0, max_value=150.0, value=65.0, step=5.0)
            fa_cost = st.slider("False Alarm Labor Cost ($ / event)", min_value=10.0, max_value=100.0, value=25.0, step=5.0)
            curr_sym = "$"

        unplanned_hours = st.slider("Historical Unplanned Downtime (hours/loom/yr)", min_value=20.0, max_value=180.0, value=75.0, step=5.0)
        detection_rate = st.slider("Acoustic Early Detection Catch Rate (%)", min_value=40, max_value=95, value=78, step=1) / 100.0

        # Calculations
        gross_loss_per_loom = unplanned_hours * downtime_cost
        prevented_savings_per_loom = detection_rate * gross_loss_per_loom
        system_overhead_per_loom = sensor_cost + (4.0 * fa_cost)
        net_saving_per_loom = prevented_savings_per_loom - system_overhead_per_loom

        fleet_gross_loss = gross_loss_per_loom * num_looms
        fleet_gross_savings = prevented_savings_per_loom * num_looms
        fleet_total_system_cost = system_overhead_per_loom * num_looms
        fleet_net_savings = net_saving_per_loom * num_looms
        roi_pct = (fleet_net_savings / max(1.0, fleet_total_system_cost)) * 100.0

    with col2:
        st.markdown("### 📈 Projected Financial Returns")

        if is_inr:
            # Format in Indian numbering system (Lakhs / Crores)
            if fleet_net_savings >= 10000000:
                net_disp = f"₹{fleet_net_savings/10000000:.2f} Crores / yr"
            else:
                net_disp = f"₹{fleet_net_savings/100000:.2f} Lakhs / yr"

            if fleet_gross_savings >= 10000000:
                gross_disp = f"₹{fleet_gross_savings/10000000:.2f} Crores"
            else:
                gross_disp = f"₹{fleet_gross_savings/100000:.2f} Lakhs"

            cost_disp = f"₹{fleet_total_system_cost:,.0f}"
        else:
            net_disp = f"${fleet_net_savings:,.0f} / yr"
            gross_disp = f"${fleet_gross_savings:,.0f}"
            cost_disp = f"${fleet_total_system_cost:,.0f}"

        r1, r2 = st.columns(2)
        r1.metric("Net Annual Mill Savings", net_disp, f"+{roi_pct:,.0f}% ROI")
        r2.metric("Return on Investment (ROI)", f"{roi_pct:,.0f}%")

        r3, r4 = st.columns(2)
        r3.metric("Prevented Downtime Value", gross_disp)
        r4.metric("Total System Hardware Cost", cost_disp)

        # Sensitivity Matrix
        st.markdown("### 🔍 Sensitivity Grid: Detection Rate vs Downtime Cost")
        rates = [0.50, 0.60, 0.70, 0.80, 0.90]
        if is_inr:
            costs = [5000.0, 8000.0, 10000.0, 15000.0, 20000.0]
            col_labels = [f"₹{int(c/1000)}k/hr" for c in costs]
        else:
            costs = [60.0, 90.0, 120.0, 150.0, 200.0]
            col_labels = [f"${int(c)}/hr" for c in costs]

        grid_matrix = []
        for r in rates:
            row_vals = []
            for c in costs:
                val = (r * (unplanned_hours * c) - system_overhead_per_loom) * num_looms
                if is_inr:
                    row_vals.append(round(val / 100000.0, 1))  # In Lakhs
                else:
                    row_vals.append(round(val, 0))
            grid_matrix.append(row_vals)

        sens_title = "Net Savings in ₹ Lakhs" if is_inr else "Net Savings in $"
        fig_sens = px.imshow(
            grid_matrix,
            labels=dict(x="Downtime Cost per Hour", y="Fault Catch Rate", color=sens_title),
            x=col_labels,
            y=[f"{int(r*100)}% Catch" for r in rates],
            color_continuous_scale="Viridis",
            text_auto=True,
        )
        fig_sens.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"),
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_sens, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("🧵 LoomEar v2.0 | Cyber-Industrial AI Edition")
