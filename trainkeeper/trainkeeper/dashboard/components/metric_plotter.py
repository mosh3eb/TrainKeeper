"""
Metric Plotter Component

Compare and visualize metrics across multiple experiments with interactive Plotly charts.
"""

import streamlit as st
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


def _load_experiment_metrics(exp_path):
    """Load metrics from an experiment directory"""
    metrics_file = exp_path / "metrics.json"
    if not metrics_file.exists():
        return None
    
    try:
        return json.loads(metrics_file.read_text())
    except:
        return None


def _get_experiment_list(artifacts_path):
    """Get list of all experiments with their IDs and timestamps"""
    experiments = []
    
    for exp_dir in sorted(artifacts_path.glob("exp-*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not exp_dir.is_dir():
            continue
        
        exp_file = exp_dir / "experiment.yaml"
        if not exp_file.exists():
            exp_file = exp_dir / "run.json"
        
        if not exp_file.exists():
            continue
        
        try:
            if exp_file.suffix == ".json":
                data = json.loads(exp_file.read_text())
            else:
                import yaml
                data = yaml.safe_load(exp_file.read_text())
            
            exp_id = data.get("exp_id", exp_dir.name)
            timestamp = data.get("environment", {}).get("timestamp", "N/A")
            
            experiments.append({
                "exp_id": exp_id,
                "path": exp_dir,
                "timestamp": timestamp,
                "display_name": f"{exp_id} ({timestamp[:19] if timestamp != 'N/A' else 'N/A'})"
            })
        except:
            continue
    
    return experiments


def render_metric_plotter(artifacts_path):
    """Render the metric comparison and plotting view"""
    
    st.markdown("## 📈 Metric Comparison")
    st.markdown("Compare metrics across multiple experiments with interactive visualizations.")
    
    # Get experiment list
    experiments = _get_experiment_list(artifacts_path)
    
    if not experiments:
        st.warning("⚠️ No experiments found. Run training experiments first.")
        return
    
    # Experiment selection
    st.markdown("### 🎯 Select Experiments to Compare")
    
    selected_exp_names = st.multiselect(
        "Choose experiments",
        options=[exp["display_name"] for exp in experiments],
        default=[experiments[0]["display_name"]] if experiments else [],
        help="Select 1 or more experiments to compare"
    )
    
    if not selected_exp_names:
        st.info("👆 Select at least one experiment to visualize metrics.")
        return
    
    # Get selected experiment paths
    selected_experiments = [
        exp for exp in experiments
        if exp["display_name"] in selected_exp_names
    ]
    
    # Load metrics for selected experiments
    exp_metrics = {}
    for exp in selected_experiments:
        metrics = _load_experiment_metrics(exp["path"])
        if metrics:
            exp_metrics[exp["exp_id"]] = metrics
    
    if not exp_metrics:
        st.warning("⚠️ No metrics found for selected experiments.")
        st.info("Make sure your training code returns a dictionary of metrics.")
        st.code("""
@run_reproducible()
def train():
    # Your training code
    return {"accuracy": 0.95, "loss": 0.05}
        """, language="python")
        return
    
    # Find common metrics across experiments
    all_metric_keys = set()
    for metrics in exp_metrics.values():
        if isinstance(metrics, dict):
            all_metric_keys.update(metrics.keys())
    
    if not all_metric_keys:
        st.warning("No comparable metrics found.")
        return
    
    # Metric visualization mode
    st.markdown("---")
    st.markdown("### 📊 Visualization")
    
    viz_mode = st.radio(
        "Visualization Mode",
        ["Bar Chart", "Heatmap", "Detailed Table"],
        horizontal=True,
        help="Choose how to visualize the metrics"
    )
    
    if viz_mode == "Bar Chart":
        # Metric selector for bar chart
        selected_metrics = st.multiselect(
            "Select metrics to plot",
            options=sorted(all_metric_keys),
            default=sorted(all_metric_keys)[:3] if len(all_metric_keys) >= 3 else sorted(all_metric_keys),
            help="Choose which metrics to display"
        )
        
        if not selected_metrics:
            st.info("Select at least one metric to plot.")
            return
        
        # Create bar chart
        for metric_name in selected_metrics:
            st.markdown(f"#### {metric_name}")
            
            # Prepare data
            exp_ids = []
            values = []
            
            for exp_id, metrics in exp_metrics.items():
                if metric_name in metrics:
                    exp_ids.append(exp_id)
                    val = metrics[metric_name]
                    # Handle different value types
                    if isinstance(val, (int, float)):
                        values.append(val)
                    else:
                        values.append(0)
            
            # Create Plotly bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=exp_ids,
                    y=values,
                    text=[f"{v:.4f}" if isinstance(v, float) else str(v) for v in values],
                    textposition='auto',
                    marker=dict(
                        color=values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=metric_name)
                    )
                )
            ])
            
            fig.update_layout(
                title=f"{metric_name} Comparison",
                xaxis_title="Experiment ID",
                yaxis_title=metric_name,
                height=400,
                template="plotly_white",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_mode == "Heatmap":
        # Create heatmap of all metrics across experiments
        st.markdown("#### Metrics Heatmap")
        
        # Prepare data for heatmap
        heatmap_data = []
        exp_ids_ordered = []
        metric_names_ordered = sorted(all_metric_keys)
        
        for exp_id, metrics in exp_metrics.items():
            row = []
            for metric_name in metric_names_ordered:
                val = metrics.get(metric_name, None)
                if isinstance(val, (int, float)):
                    row.append(val)
                else:
                    row.append(None)
            heatmap_data.append(row)
            exp_ids_ordered.append(exp_id)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=metric_names_ordered,
            y=exp_ids_ordered,
            colorscale='RdYlGn',
            text=[[f"{val:.4f}" if val is not None else "N/A" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Value")
        ))
        
        fig.update_layout(
            title="All Metrics Heatmap",
            xaxis_title="Metric",
            yaxis_title="Experiment ID",
            height=max(400, len(exp_ids_ordered) * 50),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Detailed Table
        # Create detailed comparison table
        st.markdown("#### Detailed Metrics Table")
        
        # Prepare dataframe
        table_data = []
        for exp_id, metrics in exp_metrics.items():
            row = {"Experiment ID": exp_id}
            for metric_name in sorted(all_metric_keys):
                val = metrics.get(metric_name, "N/A")
                if isinstance(val, float):
                    row[metric_name] = f"{val:.6f}"
                elif isinstance(val, int):
                    row[metric_name] = str(val)
                else:
                    row[metric_name] = str(val)
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="trainkeeper_metrics_comparison.csv",
            mime="text/csv"
        )
    
    # Statistical summary
    st.markdown("---")
    st.markdown("### 📊 Statistical Summary")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown("**Metric Statistics**")
        
        # Calculate stats for numeric metrics
        stats_data = []
        for metric_name in sorted(all_metric_keys):
            values = []
            for metrics in exp_metrics.values():
                val = metrics.get(metric_name)
                if isinstance(val, (int, float)):
                    values.append(val)
            
            if values:
                stats_data.append({
                    "Metric": metric_name,
                    "Mean": f"{sum(values)/len(values):.6f}",
                    "Min": f"{min(values):.6f}",
                    "Max": f"{max(values):.6f}",
                    "Std": f"{pd.Series(values).std():.6f}" if len(values) > 1 else "N/A"
                })
        
        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
        else:
            st.info("No numeric metrics for statistical analysis.")
    
    with summary_col2:
        st.markdown("**Best Performing**")
        
        # Find best experiment for each metric
        best_data = []
        for metric_name in sorted(all_metric_keys):
            best_exp = None
            best_val = None
            
            for exp_id, metrics in exp_metrics.items():
                val = metrics.get(metric_name)
                if isinstance(val, (int, float)):
                    if best_val is None or val > best_val:  # Assuming higher is better
                        best_val = val
                        best_exp = exp_id
            
            if best_exp:
                best_data.append({
                    "Metric": metric_name,
                    "Best Experiment": best_exp,
                    "Value": f"{best_val:.6f}" if isinstance(best_val, float) else str(best_val)
                })
        
        if best_data:
            st.dataframe(pd.DataFrame(best_data), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_metric_plotter(Path("artifacts"))
