"""
Data Drift Visualizer Component

Visualize data drift, schema changes, and data quality issues.
"""

import streamlit as st
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px


def render_drift_visualizer(artifacts_path):
    """Render data drift visualization view"""
    
    st.markdown("## 🌊 Data Drift Analysis")
    st.markdown("Analyze data drift, schema changes, and data quality across experiments.")
    
    # Look for data check files
    datacheck_files = []
    for exp_dir in artifacts_path.glob("exp-*"):
        for dc_file in exp_dir.glob("*datacheck*.json"):
            datacheck_files.append({
                "path": dc_file,
                "exp_id": exp_dir.name,
                "filename": dc_file.name
            })
    
    if not datacheck_files:
        st.info("ℹ️ No data quality checks found.")
        st.markdown("""
        **How to use DataCheck:**
        
        ```python
        from trainkeeper.datacheck import DataCheck
        
        # In your training code
        dc = DataCheck.from_dataframe(train_df)
        dc.infer_schema()
        dc.save("datacheck_baseline.json")
        
        # Later, validate new data
        issues = dc.validate(new_df)
        if issues:
            print(f"Found {len(issues)} issues!")
        ```
        """)
        return
    
    st.success(f"✅ Found {len(datacheck_files)} data check files across experiments")
    
    # File selector
    selected_file = st.selectbox(
        "Select Data Check File",
        options=[f"{dc['exp_id']}/{dc['filename']}" for dc in datacheck_files],
        help="Choose a data check file to analyze"
    )
    
    if not selected_file:
        return
    
    # Load selected file
    selected_dc = next(dc for dc in datacheck_files if f"{dc['exp_id']}/{dc['filename']}" == selected_file)
    
    try:
        data = json.loads(selected_dc["path"].read_text())
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        return
    
    # Display schema info
    st.markdown("### 📋 Schema Information")
    
    schema = data.get("schema", {})
    snapshot = data.get("snapshot", {})
    
    if not schema:
        st.warning("No schema information found in this file.")
        return
    
    # Count columns by type
    numeric_cols = sum(1 for v in schema.values() if isinstance(v, dict) and v.get("type") == "numeric")
    categorical_cols = sum(1 for v in schema.values() if isinstance(v, dict) and v.get("type") == "categorical")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Columns", len([k for k in schema.keys() if k != "_meta"]))
    
    with col2:
        st.metric("Numeric", numeric_cols)
    
    with col3:
        st.metric("Categorical", categorical_cols)
    
    with col4:
        fingerprint = schema.get("_meta", {}).get("fingerprint", "N/A")
        st.metric("Fingerprint", fingerprint[:8] if fingerprint != "N/A" else "N/A")
    
    # Column details
    st.markdown("---")
    st.markdown("### 📊 Column Details")
    
    for col_name, col_schema in schema.items():
        if col_name == "_meta":
            continue
        
        if not isinstance(col_schema, dict):
            continue
        
        with st.expander(f"📂 {col_name} ({col_schema.get('type', 'unknown')})"):
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("**Schema**")
                st.json(col_schema)
            
            with detail_col2:
                st.markdown("**Snapshot Visualization**")
                
                col_snapshot = snapshot.get(col_name, {})
                
                if col_snapshot.get("type") == "numeric":
                    # Plot histogram
                    hist = col_snapshot.get("hist", [])
                    edges = col_snapshot.get("edges", [])
                    
                    if hist and edges:
                        # Create bin centers for plotting
                        bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=bin_centers,
                                y=hist,
                                marker=dict(
                                    color=hist,
                                    colorscale='Viridis',
                                    showscale=False
                                ),
                                name=col_name
                            )
                        ])
                        
                        fig.update_layout(
                            title=f"{col_name} Distribution",
                            xaxis_title="Value",
                            yaxis_title="Count",
                            height=300,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No histogram data available")
                
                elif col_snapshot.get("type") == "categorical":
                    # Plot category counts
                    counts = col_snapshot.get("counts", {})
                    
                    if counts:
                        # Sort by count
                        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                        categories = [item[0] for item in sorted_items[:20]]  # Top 20
                        values = [item[1] for item in sorted_items[:20]]
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=categories,
                                y=values,
                                marker=dict(
                                    color=values,
                                    colorscale='Plasma',
                                    showscale=False
                                ),
                                name=col_name
                            )
                        ])
                        
                        fig.update_layout(
                            title=f"{col_name} Top Categories",
                            xaxis_title="Category",
                            yaxis_title="Count",
                            height=300,
                            template="plotly_white",
                            xaxis_tickangle=-45
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No category data available")
    
    # Data quality tips
    st.markdown("---")
    st.markdown("### 💡 Data Quality Tips")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea22 0%, #764ba244 100%); 
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>
        <h4 style='margin-top: 0;'>Best Practices</h4>
        <ul>
            <li><strong>Track baselines</strong>: Save a datacheck from your training set as a baseline</li>
            <li><strong>Validate regularly</strong>: Check new data against the baseline before training</li>
            <li><strong>Set thresholds</strong>: Configure drift thresholds based on your use case</li>
            <li><strong>Monitor in production</strong>: Use datacheck for incoming production data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_drift_visualizer(Path("artifacts"))
