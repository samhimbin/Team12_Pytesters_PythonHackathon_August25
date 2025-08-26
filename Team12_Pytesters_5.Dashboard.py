import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates


# ============================
# Dashboard Layout
# ============================
st.set_page_config(page_title="CGM Type1 Diabetes Dashboard", layout="wide")

# Title & Intro
st.title("CGM Type1 Diabetes Data Analysis Dashboard")
st.write("""
This dashboard demonstrates:
- Data Cleanup  
- Descriptive Insights  
- Prescriptive Recommendations  
- Predictive Analysis
""")

# ============================
# Load Data
# ============================
df = pd.read_csv("mergedraw_file.csv")
merged_df = pd.read_csv("merged_cleandata.csv")
demographic_df = pd.read_csv("demographic_cleandata.csv")

# ============================
# Tabs for Navigation
# ============================
tab1, tab2, tab3, tab4 = st.tabs(["Data Cleanup", "Descriptive", "Prescriptive", "Predictive"])

# ============================
# TAB 1: Data Cleanup
# ============================
with tab1:
    st.header("1. Data Cleanup")

    st.markdown("We combined **25 patients’ files** together for easier group analysis.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Data")
        st.write(f"Shape: {df.shape}")
        st.dataframe(df.head())
        st.info("""
        Observations:  
        - Some patients had **hundreds of thousands of records** (spanning a year).  
        - Some patients had **fewer than 14 days** of data (as low as 9 days).  
        """)

    with col2:
        st.subheader("Cleaned Data")
        st.write(f"Shape: {merged_df.shape}")
        st.dataframe(merged_df.head())
        st.success("""
        ✔ Standardized all patients to **14 days** of data  
        ✔ Negative Bolus values set to **zero**  
        ✔ Applied **range checks** for glucose, heart rate, and calories burned  
        """)

    st.markdown("---")
    st.subheader("Cleanup Summary")
    st.markdown("""
    - **Filtered data** to 14-day sample set per patient  
    - **Corrected Bolus values** (negative → zero)  
    - **Validated ranges** for glucose, heart rate, calories burned  
    """)
    st.info("Data is now standardized across patients and ready for reliable analysis.")


# ============================
# TAB 2: Descriptive Analysis
# ============================
with tab2:
    st.header("2. Descriptive Analysis")

    # -------------------
    # Question 1: Rolling HR
    # -------------------
    st.subheader("Q1: Rolling averages of heart rate during high vs. low glucose per patient")

    # Ensure datetime and sort
    merged_df["time"] = pd.to_datetime(merged_df["time"])
    merged_df = merged_df.sort_values(["patient_id", "time"])

    # Rolling 6-hour average HR
    merged_df["hr_rolling6h"] = (
        merged_df.groupby("patient_id", group_keys=False)[["time", "heart_rate"]]
          .apply(lambda g: (
              g.set_index("time")["heart_rate"]
               .rolling("6h", min_periods=1)
               .mean()
               .reset_index(drop=True)
          ))
          .reset_index(drop=True)
    )

    # Summary table
    summary_hr = (
        merged_df.groupby(["patient_id", "glucose_range_level"])["hr_rolling6h"]
          .mean()
          .unstack(fill_value=0)
    )

    st.write("###  Summary Table of 6-hour Rolling HR by Glucose Range Level")

    # Plot
    fig_hr, ax_hr = plt.subplots(figsize=(12,6))
    summary_hr.plot(kind="bar", ax=ax_hr)
    ax_hr.set_title("Patient-wise Avg 6H Rolling HR by Glucose Range Level")
    ax_hr.set_ylabel("Heart Rate (bpm)")
    ax_hr.set_xlabel("Patient ID")
    ax_hr.set_xticklabels(ax_hr.get_xticklabels(), rotation=45, ha="right")
    ax_hr.grid(axis="y", linestyle="--", alpha=0.7)
    ax_hr.legend(title="Glucose Range Level")
    st.pyplot(fig_hr)

    # -------------------
    # Question 2: Post-meal glucose
    # -------------------
    st.subheader("Q2: Post-meal glucose change over 2 hours and insulin effect")

    # Collect 2-hour glucose windows
    meal_windows = []
    for pid, pdf in merged_df.groupby("patient_id"):
        pdf = pdf.sort_values("time")
        meal_times = pdf.loc[pdf["carb_input"] > 0, "time"]

        for mt in meal_times:
            window = pdf[(pdf["time"] >= mt) & (pdf["time"] <= mt + pd.Timedelta(hours=2))].copy()
            if len(window) > 0:
                start_glucose = window["glucose"].iloc[0]
                window["glucose_change"] = window["glucose"] - start_glucose
                window["minutes"] = (window["time"] - mt).dt.total_seconds() / 60
                window["patient_id"] = pid
                meal_windows.append(window[["patient_id", "minutes", "glucose_change", "glucose_range_level"]])

    patient_trajs = pd.concat(meal_windows)

    # Compute avg glucose change at ~120 min
    overall_stats = []
    for pid, traj_df in patient_trajs.groupby("patient_id"):
        avg_120 = traj_df.loc[traj_df["minutes"].between(110, 130), "glucose_change"].mean()
        range_counts = traj_df.loc[traj_df["minutes"].between(110, 130), "glucose_range_level"].value_counts(normalize=True)
        overall_stats.append({
            "patient_id": pid,
            "avg_glucose_rise_120min": avg_120,
            "pct_below_range": range_counts.get("Below Range", 0) * 100,
            "pct_in_range": range_counts.get("In Range", 0) * 100,
            "pct_above_range": range_counts.get("Above Range", 0) * 100,
        })

    overall_df = pd.DataFrame(overall_stats)

    # Plot avg glucose rise
    fig_glucose, ax_glucose = plt.subplots(figsize=(12, 6))
    ax_glucose.bar(overall_df["patient_id"], overall_df["avg_glucose_rise_120min"], color="skyblue")
    ax_glucose.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_glucose.set_ylabel("Avg Glucose Change at 120 min (mg/dL)")
    ax_glucose.set_title("Average Post-Meal Glucose Change (120 min) per Patient")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_glucose)

    # Summary table
    st.markdown("###  Summary of Glucose Range Distribution (~120 min post-meal)")
    st.dataframe(overall_df[["patient_id", "pct_below_range", "pct_in_range", "pct_above_range"]])

    st.markdown("###  Detailed Patient Results")
    st.dataframe(overall_df.sort_values("avg_glucose_rise_120min"))


# --- Prescriptive Analysis Tab ---
with tab3:
    st.header("3. Prescriptive Analysis")

    st.subheader("Q1: Longer Hypoglycemia Incidents and Carb Recommendations")
    st.write("""
    Identify which patients experience longer hypoglycemia events and when they should take carbs.
    """)

    # Ensure date column exists
    merged_df['date'] = pd.to_datetime(merged_df['time']).dt.date

    # Create below_range flag
    merged_df['below_range'] = merged_df['glucose_range_level'].str.strip().str.lower().apply(
        lambda x: 1 if x == 'below range' else 0
    )

    # Hypoglycemia rows finder
    def consecutive_hypo_rows(group, min_length=3):
        group = group.copy()
        group['event_group'] = (group['below_range'] != group['below_range'].shift()).cumsum()
        hypo_events = group[group['below_range'] == 1].groupby('event_group')
        result = []
        for eid, rows in hypo_events:
            if len(rows) >= min_length:
                duration_intervals = len(rows)
                num_blocks = duration_intervals // 3
                result.append({
                    'patient_id': rows['patient_id'].iloc[0],
                    'date': rows['date'].iloc[0],
                    'event_id': eid,
                    'start_time': rows['time'].iloc[0],
                    'end_time': rows['time'].iloc[-1],
                    'duration_intervals': duration_intervals*5,
                    'recommendation': f"Take carbs every 15 min ({num_blocks} times)" if num_blocks > 0 else "Monitor only"
                })
        return pd.DataFrame(result)

    hypo_events = merged_df.groupby(['patient_id', 'date']).apply(consecutive_hypo_rows).reset_index(drop=True)

    st.dataframe(hypo_events)

    # Timeline chart for hypoglycemia events
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, row in hypo_events.iterrows():
        ax.barh(row['patient_id'], 
                (row['end_time'] - row['start_time']).total_seconds()/60, 
                left=row['start_time'], 
                height=0.3, 
                label=row['recommendation'] if i==0 else "")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.xlabel("Time")
    plt.ylabel("Patient ID")
    plt.title("Hypoglycemia Events Timeline per Patient")
    plt.tight_layout()
    st.pyplot(fig)
    st.info("""
    **What the chart shows:**  
    - Bars represent hypoglycemia incidents.  
    - Start point = when glucose first dropped below the safe range.  
    - End point = when glucose returned to normal.  
    - Length = duration of the event (longer bars = more severe incidents).  

    **Key Insight:**  
    - You can quickly see which patients experience short vs. long episodes.  

    **Examples:**  
    - If one patient has longer bars, they’re at higher risk and need earlier carb interventions.  
    - If another patient has mostly short events, their glucose usually recovers faster.
    """)
    # --- Q2: Hyperglycemia Events and Duration ---
    st.subheader("Q2: Hyperglycemia Events and Average Duration")
    st.write("Identify hyperglycemia events and their duration for each patient.")

    # Create above_range flag
    merged_df['above_range'] = merged_df['glucose_range_level'].str.strip().str.lower().apply(
        lambda x: 1 if x == 'above range' else 0
    )

    # Hyperglycemia rows finder
    def consecutive_hyper_rows(group, min_length=3):
        group = group.copy()
        group['event_group'] = (group['above_range'] != group['above_range'].shift()).cumsum()
        hyper_events = group[group['above_range'] == 1].groupby('event_group')
        result = []
        for eid, rows in hyper_events:
            if len(rows) >= min_length:
                duration_intervals = len(rows)
                num_blocks = duration_intervals // 3
                result.append({
                    'patient_id': rows['patient_id'].iloc[0],
                    'date': rows['date'].iloc[0],
                    'event_id': eid,
                    'start_time': rows['time'].iloc[0],
                    'end_time': rows['time'].iloc[-1],
                    'duration_intervals': duration_intervals*5,
                    'recommendation': f"Take corrective action every 15 min ({num_blocks} times)" if num_blocks > 0 else "Monitor only"
                })
        return pd.DataFrame(result)

    hyper_events = merged_df.groupby(['patient_id', 'date']).apply(consecutive_hyper_rows).reset_index(drop=True)

    # Plots: Events count and duration
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    merged_df['patient_id'].value_counts().plot(kind='bar', ax=axes[0])
    axes[0].set_title("Number of Hyperglycemia Events per Patient")
    axes[0].set_xlabel("Patient ID")
    axes[0].set_ylabel("Number of Events")
    axes[0].tick_params(axis='x', rotation=90)

    hyper_events.groupby('patient_id')['duration_intervals'].mean().plot(kind='bar', ax=axes[1])
    axes[1].set_title("Average Hyperglycemia Event Duration per Patient")
    axes[1].set_xlabel("Patient ID")
    axes[1].set_ylabel("Avg Duration (minutes)")
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Key Insights: Hyperglycemia Events")

    st.markdown("""
    - **Event Duration Varies:** Some patients experience short events (~30–50 min), while others have prolonged episodes (>4 hours).  
    - **Multiple Interventions Needed:** Single corrective actions are often insufficient; repeated interventions may be required.  
    - **Patient-Specific Patterns:** Certain patients (e.g., HUPA0001P) show multiple long-duration events on the same day, highlighting individualized management.  
    - **Overall Hyperglycemia Burden:** Summing event durations per patient quantifies total high-glucose exposure, critical for risk assessment.
    """)

    st.info("Some patients have short events (~30–50 min), while others experience prolonged hyperglycemia (>4 hours). Repeated interventions may be needed, and patient-specific patterns emerge. Total duration per patient helps assess overall hyperglycemia burden.")



    # --- Q3: Hyper vs Hypo Comparison ---
    st.subheader("Q3: Patient-wise Hyper vs Hypo Glycemia Summary")
    st.write("Classify patients as more affected by hyperglycemia, hypoglycemia, or equally affected by both.")

    # Daily aggregates
    hypo_daily = hypo_events.groupby(['patient_id', 'date']).size().reset_index(name='hypo_events')
    hyper_daily = hyper_events.groupby(['patient_id', 'date']).size().reset_index(name='hyper_events')

    daily_summary = pd.merge(hypo_daily, hyper_daily, on=['patient_id', 'date'], how='outer').fillna(0)
    daily_summary['hypo_events'] = daily_summary['hypo_events'].astype(int)
    daily_summary['hyper_events'] = daily_summary['hyper_events'].astype(int)

    # Patient-level summary
    patient_summary = daily_summary.groupby("patient_id")[["hyper_events", "hypo_events"]].sum().reset_index()

    def classify(row):
        if row['hyper_events'] > row['hypo_events']:
            return "More Hyperglycemia"
        elif row['hypo_events'] > row['hyper_events']:
            return "More Hypoglycemia"
        else:
            return "Equal"

    patient_summary['status'] = patient_summary.apply(classify, axis=1)
    st.dataframe(patient_summary)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10,8))
    colors = {"More Hyperglycemia":"red", "More Hypoglycemia":"blue", "Equal":"green"}
    for _, row in patient_summary.iterrows():
        ax.scatter(row['hyper_events'], row['hypo_events'], 
                    color=colors[row['status']], s=100)
        ax.text(row['hyper_events']+0.2, row['hypo_events']+0.2, row['patient_id'], fontsize=8)
    max_val = max(patient_summary['hyper_events'].max(), patient_summary['hypo_events'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, label="Equal Line")
    ax.set_xlabel("Total Hyperglycemia Events")
    ax.set_ylabel("Total Hypoglycemia Events")
    ax.set_title("Patient-wise Summary: Hyper vs Hypo Glycemia Events")
    ax.legend(colors.keys())
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("## Key Insights: Glycemic Event Patterns")

    st.markdown("""
    - **Hyperglycemia-prone patients:** Most patients (e.g., HUPA0001P, HUPA0003P, HUPA0004P) frequently have glucose spikes above the target range.  
    - **Hypoglycemia-prone patients:** A smaller group (e.g., HUPA0002P, HUPA0016P, HUPA0018P) experience dangerously low glucose events.  
    - **No balanced patients:** All patients tend to lean toward one type of glycemic risk rather than being equally affected.
""")

    st.markdown("## Recommended Actions")

    st.markdown("""
    **For hyperglycemia-prone patients:**  
    - Adjust insulin doses  
    - Monitor meals and carbohydrate intake more closely  
    - Implement lifestyle interventions to reduce glucose spikes  

    **For hypoglycemia-prone patients:**  
    - Implement preventive monitoring  
    - Reduce insulin or have fast-acting glucose ready for emergencies  

    **Priority:** Patients with the largest imbalance may be at higher risk of complications and should be monitored more closely.
""")


    st.info("This analysis helps prioritize interventions based on patient-specific glycemic risk patterns.")

    
with tab4:
    
    st.header("4. Predictive Analysis")
    st.subheader("Hypoglycemia Early Warning Prediction")

    st.write("""
    We explored basal insulin adequacy and early warning of hypoglycemia:  
    - Flags glucose < 70 mg/dL as hypoglycemia  
    - Computes heart rate-based features to predict 15-min ahead hypo events  
    """)

    # Merge with demographic info if available
    combined_df = merged_df.merge(demographic_df, on='patient_id', how='left')

    # Step 1: Sort and prepare
    combined_df = combined_df.sort_values(['patient_id', 'time'])
    combined_df['time'] = pd.to_datetime(combined_df['time'])

    # Step 2: Flag hypoglycemia events (<70 mg/dL)
    threshold = 70
    combined_df['hypo_event'] = (combined_df['glucose'] < threshold).astype(int)

    # Step 3: Create HR-based predictive features
    window = '10min'
    features = []

    for pid, group in combined_df.groupby('patient_id'):
        g = group.set_index('time')
        g['hr_mean_10m'] = g['heart_rate'].rolling(window=window).mean()
        g['hr_std_10m'] = g['heart_rate'].rolling(window=window).std()
        g['hr_delta'] = g['heart_rate'].diff()

        # Mark windows 15 minutes before hypo events
        g['hypo_ahead_15m'] = (
            g['hypo_event']
            .shift(-1)
            .rolling('15min')
            .max()
            .fillna(0)
            .astype(int)
        )
        features.append(g.reset_index())

    feature_df = pd.concat(features).sort_values(['patient_id', 'time'])

    # Step 4: Summary of predicted hypo windows per patient
    summary = feature_df.groupby('patient_id')['hypo_ahead_15m'].sum().reset_index()
    summary = summary.rename(columns={'hypo_ahead_15m': 'predicted_hypo_windows'})

    st.markdown("### 📊 Predicted 15-min Hypoglycemia Windows by Patient")

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(summary['patient_id'], summary['predicted_hypo_windows'], color='salmon')
    ax.set_xlabel('Patient ID')
    ax.set_ylabel('15-min Hypo Prediction Windows')
    ax.set_title('Potential Early Warning Opportunities by Patient')
    ax.set_xticklabels(summary['patient_id'], rotation=90, ha='center')
    plt.tight_layout()

    st.pyplot(fig)

    st.info("This visualization highlights patients with the highest number of predicted early warning windows for hypoglycemia. Useful for proactive insulin adjustments and monitoring.")

