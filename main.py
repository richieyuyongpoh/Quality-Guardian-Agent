# Quality Guardian Agent - A Streamlit Demonstration

import streamlit as st
import pandas as pd
import numpy as np
import random

# --- App Configuration ---
st.set_page_config(
    page_title="Quality Guardian Agent",
    page_icon="🛡️",
    layout="wide"
)

# --- App Header ---
st.title("🛡️ Quality Guardian Agent - Live Demo")
st.markdown("""
This application demonstrates the core functionality of the **Quality Guardian Agent**.

The agent's goal is to prevent defective products from reaching customers by moving beyond simple pass/fail tests. It achieves this by:
1.  **Perceiving** a holistic set of sensor data from each manufactured unit.
2.  **Reasoning** to identify complex patterns and anomalies that predict future failure, even when individual readings are within tolerance.
3.  **Acting** by flagging at-risk units for rework *before* they leave the factory.

*This demo is based on the concepts presented in the AI Transformation Workshop.*
""")

# --- Attribution and License ---
st.sidebar.title("About")
st.sidebar.markdown("""
**Designed by Richie, TAR UMT**
<a href="mailto:yuyp@tarc.edu.my">yuyp@tarc.edu.my</a>
""", unsafe_allow_html=True)
st.sidebar.info("""
**Disclaimer:** This is a demonstration application for illustrative purposes only. The data is simulated.
""")
st.sidebar.markdown("""
**License:** This work is licensed under a
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
<br/><br/>
<a rel="license" href="http://creativecommons.org/l/by-nc/4.0/88x31.png" />
""", unsafe_allow_html=True)


# --- Data Simulation & Constants ---
COST_OF_ESCAPE = 2500  # Cost of a field failure (warranty, shipping, brand damage)
COST_OF_REWORK = 300   # Cost to fix a unit identified before shipping

def get_production_batch(batch_size=50):
    """Creates a DataFrame simulating a production batch with sensor readings."""
    data = []
    num_defects = random.randint(2, 4)
    defect_indices = random.sample(range(batch_size), num_defects)

    for i in range(batch_size):
        is_defective = i in defect_indices
        
        # A "hidden" defect has multiple borderline readings, but none are out of spec individually
        temp = random.uniform(75.0, 85.0) if is_defective else random.uniform(65.0, 78.0)
        volt = random.uniform(1.15, 1.20) if is_defective else random.uniform(1.21, 1.25)
        solder_integrity = random.uniform(98.0, 99.0) if is_defective else random.uniform(99.1, 99.9)

        data.append({
            'Unit_ID': f"SN-B4-{1000+i}",
            'Core_Temp_C': round(temp, 2),
            'Voltage_V': round(volt, 2),
            'Solder_Integrity_%': round(solder_integrity, 2),
            'Actual_Status': 'Defective' if is_defective else 'Good'
        })
    return pd.DataFrame(data)

# --- Agent & Inspection Logic ---
def run_standard_inspection(df):
    """Simulates a standard inspection based on simple thresholds."""
    df['Standard_Inspection'] = 'Pass'
    # Standard inspection has wider tolerances and misses the subtle, combined issues.
    df.loc[(df['Core_Temp_C'] > 88.0) | (df['Voltage_V'] < 1.10), 'Standard_Inspection'] = 'Fail'
    return df

def run_quality_guardian_agent(df):
    """The agent uses a more sophisticated model to find hidden defects."""
    # The agent's "reasoning" is to calculate an anomaly score.
    # It flags units with multiple borderline readings.
    df['Anomaly_Score'] = 0
    df.loc[df['Core_Temp_C'] > 75, 'Anomaly_Score'] += 1
    df.loc[df['Voltage_V'] < 1.20, 'Anomaly_Score'] += 1
    df.loc[df['Solder_Integrity_%'] < 99.0, 'Anomaly_Score'] += 2 # Solder is critical
    
    df['Agent_Verdict'] = 'Pass'
    df.loc[df['Anomaly_Score'] >= 2, 'Agent_Verdict'] = 'Flag for Rework'
    
    reasoning = "Agent analyzed patterns across all sensors. It flagged units with a high 'Anomaly Score', indicating a combination of risk factors, even if no single reading failed its individual spec. This predicts future failures."
    return df, reasoning

# --- Main App UI ---
if 'batch' not in st.session_state:
    st.session_state.batch = get_production_batch()
    st.session_state.inspected_batch = None
    st.session_state.history = []

st.header("Final Quality Assurance Simulation")
st.markdown("Comparing a standard inspection process with the AI Guardian's predictive analysis.")

# --- Display Batch Data ---
with st.expander("Show Raw Production Batch Data"):
    st.dataframe(st.session_state.batch[['Unit_ID', 'Core_Temp_C', 'Voltage_V', 'Solder_Integrity_%']], use_container_width=True, hide_index=True)

st.divider()

# --- Control & Agent Log ---
st.header("Agent Control & Reasoning")
if st.button("🛡️ Run QA Inspections", type="primary"):
    standard_inspected_df = run_standard_inspection(st.session_state.batch.copy())
    # CORRECTED LINE: Fixed the typo in the function name.
    agent_inspected_df, reasoning = run_quality_guardian_agent(standard_inspected_df)
    
    st.session_state.inspected_batch = agent_inspected_df
    st.session_state.reasoning = reasoning
    
    # Calculate impact for history
    escapes = agent_inspected_df[(agent_inspected_df['Standard_Inspection'] == 'Pass') & (agent_inspected_df['Actual_Status'] == 'Defective')]
    caught_by_agent = agent_inspected_df[(agent_inspected_df['Agent_Verdict'] == 'Flag for Rework') & (agent_inspected_df['Actual_Status'] == 'Defective')]
    
    cost_of_escapes = len(escapes) * COST_OF_ESCAPE
    cost_of_rework = len(caught_by_agent) * COST_OF_REWORK
    net_value = cost_of_escapes - cost_of_rework
    
    run_number = len(st.session_state.history) + 1
    st.session_state.history.append({
        'Run': run_number,
        'Escapes Prevented': len(escapes),
        'Net Value Generated': net_value if net_value > 0 else 0
    })
    st.rerun()

if st.button("🔄 Generate New Production Batch"):
    st.session_state.batch = get_production_batch()
    st.session_state.inspected_batch = None
    st.session_state.reasoning = None
    st.session_state.history = []
    st.rerun()

if 'reasoning' in st.session_state and st.session_state.reasoning:
    st.info(f"**Agent's Reasoning:** {st.session_state.reasoning}")

st.divider()

# --- Inspection Results & Benefit Simulation ---
st.header("Inspection Results & Tangible Benefits")
if st.session_state.inspected_batch is not None:
    results_df = st.session_state.inspected_batch
    
    st.subheader("Inspection Verdicts")
    
    def highlight_results(row):
        # Default style is no style
        style = pd.Series('', index=row.index)
        # Highlight escapes: units that passed standard inspection but are actually defective
        if row['Standard_Inspection'] == 'Pass' and row['Actual_Status'] == 'Defective':
            style['Standard_Inspection'] = 'background-color: #8B0000; color: white'
        # Highlight agent catches: units the agent flagged that are actually defective
        if row['Agent_Verdict'] == 'Flag for Rework' and row['Actual_Status'] == 'Defective':
            style['Agent_Verdict'] = 'background-color: #006400; color: white'
        return style

    st.dataframe(
        results_df.style.apply(highlight_results, axis=1),
        column_config={
            "Actual_Status": None, # Hides the 'answer' column from the UI
            "Anomaly_Score": st.column_config.NumberColumn(
                "Anomaly Score",
                help="A risk score calculated by the agent. Higher scores indicate higher risk.",
                format="%d"
            )
        },
        use_container_width=True,
        hide_index=True
    )

    st.markdown("Key: <span style='background-color: #8B0000; color: white; padding: 2px;'>Defective Escape</span> | <span style='background-color: #006400; color: white; padding: 2px;'>Defect Caught by Agent</span>", unsafe_allow_html=True)

    # --- Financial Analysis ---
    st.subheader("Latest Simulation Run Analysis")
    
    escapes = results_df[(results_df['Standard_Inspection'] == 'Pass') & (results_df['Actual_Status'] == 'Defective')]
    caught_by_agent = results_df[(results_df['Agent_Verdict'] == 'Flag for Rework') & (results_df['Actual_Status'] == 'Defective')]
    
    cost_of_escapes = len(escapes) * COST_OF_ESCAPE
    cost_of_rework = len(caught_by_agent) * COST_OF_REWORK
    net_value = cost_of_escapes - cost_of_rework
    
    if len(escapes) > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Defective Units Missed by Standard Inspection", f"{len(escapes)} Units", help="These 'escapes' would have resulted in field failures and warranty claims.")
        c2.metric("Cost of Escapes (Avoided)", f"${cost_of_escapes:,.0f}", help=f"Total potential cost if these {len(escapes)} units failed in the field.")
        c3.metric("Net Value Generated by Agent", f"${net_value:,.0f}", delta=f"${net_value:,.0f}", help="The value of preventing the escapes, minus the cost of reworking the units the agent found.")
    else:
        st.success("No defective units were missed by the standard inspection in this run.")
        
    # --- Cumulative & Trend Analysis ---
    if len(st.session_state.history) > 0:
        st.subheader("Cumulative & Trend Analysis (All Runs)")
        history_df = pd.DataFrame(st.session_state.history)
        
        c4, c5 = st.columns(2)
        with c4:
            total_value = history_df['Net Value Generated'].sum()
            total_escapes = history_df['Escapes Prevented'].sum()
            st.metric("Total Net Value Generated (All Runs)", f"${total_value:,.0f}")
            st.metric("Total Escapes Prevented (All Runs)", f"{total_escapes} Units")
        with c5:
            st.markdown("**Net Value Trend**")
            st.bar_chart(history_df.set_index('Run')['Net Value Generated'])
else:
    st.info("Run the QA Inspections to see the results and financial analysis.")

