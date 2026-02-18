"""
TrainKeeper Interactive Dashboard - Main Application

A modern, beautiful Streamlit dashboard for exploring experiments,
comparing metrics, visualizing drift, and monitoring training runs.
"""

import streamlit as st
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trainkeeper.experiment import load_experiment, compare_experiments
from trainkeeper.dashboard.components.experiment_explorer import render_experiment_explorer
from trainkeeper.dashboard.components.metric_plotter import render_metric_plotter
from trainkeeper.dashboard.components.drift_visualizer import render_drift_visualizer
from trainkeeper.dashboard.components.system_monitor import render_system_monitor


def load_custom_css():
    """Load custom CSS for premium design"""
    css_path = Path(__file__).parent / "static" / "custom.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def main():
    # Page configuration
    st.set_page_config(
        page_title="TrainKeeper Dashboard",
        page_icon="🚂",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Load custom CSS for premium look
    load_custom_css()
    
    # Header
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0 2rem 0;'>
            <h1 style='font-size: 3rem; font-weight: 700; 
                       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent;
                       margin-bottom: 0.5rem;'>
                🚂 TrainKeeper Dashboard
            </h1>
            <p style='font-size: 1.2rem; color: #666; margin: 0;'>
                Reproducible, Debuggable, Efficient ML Training
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    
    # Artifacts directory selection
    artifacts_dir = st.sidebar.text_input(
        "Artifacts Directory",
        value="artifacts",
        help="Path to your TrainKeeper artifacts directory"
    )
    
    page = st.sidebar.radio(
        "Select View",
        [
            "🔍 Experiment Explorer",
            "📈 Metric Comparison",
            "🌊 Data Drift Analysis",
            "💻 System Monitor"
        ],
        index=0,
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='padding: 1rem; background: linear-gradient(135deg, #667eea22 0%, #764ba244 100%); 
                    border-radius: 0.5rem; margin-top: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0;'>💡 Quick Tips</h4>
            <ul style='margin: 0; padding-left: 1.2rem; font-size: 0.9rem;'>
                <li>Use filters to find specific experiments</li>
                <li>Click on metrics to compare trends</li>
                <li>Export reports for team sharing</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Main content area
    artifacts_path = Path(artifacts_dir)
    
    if not artifacts_path.exists():
        st.error(f"❌ Artifacts directory not found: {artifacts_dir}")
        st.info("💡 Run `tk init` to create a new project or specify an existing artifacts directory.")
        return
    
    # Route to appropriate page
    if page == "🔍 Experiment Explorer":
        render_experiment_explorer(artifacts_path)
    elif page == "📈 Metric Comparison":
        render_metric_plotter(artifacts_path)
    elif page == "🌊 Data Drift Analysis":
        render_drift_visualizer(artifacts_path)
    elif page == "💻 System Monitor":
        render_system_monitor(artifacts_path)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; font-size: 0.9rem; padding: 1rem 0;'>
            Built with ❤️ by the TrainKeeper team | 
            <a href='https://github.com/mosh3eb/TrainKeeper' target='_blank' 
               style='color: #667eea; text-decoration: none;'>GitHub</a> | 
            <a href='https://trainkeeper.readthedocs.io' target='_blank'
               style='color: #667eea; text-decoration: none;'>Docs</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
