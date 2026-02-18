"""
Experiment Explorer Component

Browse, filter, and explore all experiments with rich metadata display.
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def _load_all_experiments(artifacts_path):
    """Load metadata for all experiments"""
    experiments = []
    
    for exp_dir in sorted(artifacts_path.glob("exp-*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not exp_dir.is_dir():
            continue
            
        try:
            exp_file = exp_dir / "experiment.yaml"
            if not exp_file.exists():
                exp_file = exp_dir / "run.json"
            
            if not exp_file.exists():
                continue
                
            # Load experiment metadata
            if exp_file.suffix == ".json":
                data = json.loads(exp_file.read_text())
            else:
                import yaml
                data = yaml.safe_load(exp_file.read_text())
            
            # Load metrics if available
            metrics_file = exp_dir / "metrics.json"
            metrics = {}
            if metrics_file.exists():
                metrics = json.loads(metrics_file.read_text())
            
            experiments.append({
                "exp_id": data.get("exp_id", exp_dir.name),
                "timestamp": data.get("environment", {}).get("timestamp", "N/A"),
                "seed": data.get("seed", "N/A"),
                "entrypoint": data.get("entrypoint", {}).get("name", "N/A"),
                "metrics": metrics,
                "path": str(exp_dir),
                "data": data
            })
        except Exception as e:
            st.warning(f"⚠️ Failed to load {exp_dir.name}: {e}")
            continue
    
    return experiments


def render_experiment_explorer(artifacts_path):
    """Render the experiment explorer view"""
    
    st.markdown("## 🔍 Experiment Explorer")
    st.markdown("Browse and explore all your training experiments with detailed metadata.")
    
    # Load experiments
    with st.spinner("Loading experiments..."):
        experiments = _load_all_experiments(artifacts_path)
    
    if not experiments:
        st.warning("⚠️ No experiments found. Run a training with TrainKeeper to see experiments here.")
        st.code("tk run -- python train.py", language="bash")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Experiments",
            len(experiments),
            delta=None,
            help="Total number of experiments found"
        )
    
    with col2:
        unique_seeds = len(set(exp["seed"] for exp in experiments if exp["seed"] != "N/A"))
        st.metric(
            "Unique Seeds",
            unique_seeds,
            help="Number of different random seeds used"
        )
    
    with col3:
        with_metrics = sum(1 for exp in experiments if exp["metrics"])
        st.metric(
            "With Metrics",
            with_metrics,
            delta=f"{with_metrics/len(experiments)*100:.0f}%",
            help="Experiments with recorded metrics"
        )
    
    with col4:
        # Most recent experiment time
        try:
            recent_time = experiments[0]["timestamp"]
            if recent_time != "N/A":
                recent_dt = datetime.fromisoformat(recent_time.replace("Z", "+00:00"))
                time_ago = datetime.now(recent_dt.tzinfo) - recent_dt
                if time_ago.days > 0:
                    time_str = f"{time_ago.days}d ago"
                elif time_ago.seconds > 3600:
                    time_str = f"{time_ago.seconds // 3600}h ago"
                else:
                    time_str = f"{time_ago.seconds // 60}m ago"
                st.metric("Last Run", time_str, help="Time since last experiment")
            else:
                st.metric("Last Run", "N/A")
        except:
            st.metric("Last Run", "N/A")
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🎯 Filters")
    
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        search_query = st.text_input(
            "🔎 Search",
            placeholder="Search by exp_id, entrypoint...",
            help="Filter experiments by ID or entrypoint name"
        )
    
    with filter_col2:
        # Seed filter
        all_seeds = sorted(set(exp["seed"] for exp in experiments if exp["seed"] != "N/A"))
        seed_filter = st.multiselect(
            "Random Seed",
            options=all_seeds,
            default=None,
            help="Filter by random seed"
        )
    
    # Apply filters
    filtered_experiments = experiments
    
    if search_query:
        filtered_experiments = [
            exp for exp in filtered_experiments
            if search_query.lower() in exp["exp_id"].lower()
            or search_query.lower() in str(exp["entrypoint"]).lower()
        ]
    
    if seed_filter:
        filtered_experiments = [
            exp for exp in filtered_experiments
            if exp["seed"] in seed_filter
        ]
    
    st.markdown(f"### 📋 Experiments ({len(filtered_experiments)} found)")
    
    # Convert to dataframe for display
    if filtered_experiments:
        df_data = []
        for exp in filtered_experiments:
            # Extract key metrics
            metrics_str = ""
            if exp["metrics"]:
                if isinstance(exp["metrics"], dict):
                    metrics_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                                           for k, v in list(exp["metrics"].items())[:3])
                else:
                    metrics_str = str(exp["metrics"])
            
            df_data.append({
                "Exp ID": exp["exp_id"],
                "Timestamp": exp["timestamp"],
                "Seed": exp["seed"],
                "Entrypoint": exp["entrypoint"],
                "Metrics": metrics_str or "N/A"
            })
        
        df = pd.DataFrame(df_data)
        
        # Display with styling
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Exp ID": st.column_config.TextColumn("Exp ID", width="medium"),
                "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                "Seed": st.column_config.NumberColumn("Seed", width="small"),
                "Entrypoint": st.column_config.TextColumn("Entrypoint", width="medium"),
                "Metrics": st.column_config.TextColumn("Metrics", width="large"),
            }
        )
        
        # Detailed view expanders
        st.markdown("### 📄 Detailed View")
        
        for exp in filtered_experiments[:10]:  # Limit to first 10 for performance
            with st.expander(f"🔬 {exp['exp_id']} - {exp['entrypoint']}"):
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("**Experiment Info**")
                    st.json({
                        "exp_id": exp["exp_id"],
                        "seed": exp["seed"],
                        "timestamp": exp["timestamp"],
                        "entrypoint": exp["entrypoint"],
                        "path": exp["path"]
                    })
                
                with detail_col2:
                    st.markdown("**Metrics**")
                    if exp["metrics"]:
                        st.json(exp["metrics"])
                    else:
                        st.info("No metrics recorded")
                
                # Environment details
                if "environment" in exp["data"]:
                    st.markdown("**Environment**")
                    env = exp["data"]["environment"]
                    st.json({
                        "python_version": env.get("python_version", "N/A"),
                        "platform": env.get("platform", "N/A"),
                        "hostname": env.get("hostname", "N/A"),
                    })
    else:
        st.info("No experiments match your filters.")


if __name__ == "__main__":
    # For testing
    st.set_page_config(layout="wide")
    render_experiment_explorer(Path("artifacts"))
