"""
System Monitor Component

Real-time and historical system resource monitoring.
"""

import streamlit as st
import json
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime


def render_system_monitor(artifacts_path):
    """Render system monitoring view"""
    
    st.markdown("## 💻 System Monitor")
    st.markdown("Monitor system resources and environment across experiments.")
    
    # Load system info from experiments
    experiments_sys_info = []
    
    for exp_dir in sorted(artifacts_path.glob("exp-*"), key=lambda p: p.stat().st_mtime, reverse=True):
        system_file = exp_dir / "system.json"
        if system_file.exists():
            try:
                sys_data = json.loads(system_file.read_text())
                sys_data["exp_id"] = exp_dir.name
                experiments_sys_info.append(sys_data)
            except:
                continue
    
    if not experiments_sys_info:
        st.warning("⚠️ No system information found in experiments.")
        st.info("System info is automatically captured when using `@run_reproducible(capture_env=True)`")
        return
    
    st.success(f"✅ Found system info for {len(experiments_sys_info)} experiments")
    
    # Latest experiment overview
    st.markdown("### 🖥️ Latest Experiment System Info")
    
    latest = experiments_sys_info[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        python_ver = latest.get("python_version", "N/A")
        if python_ver != "N/A":
            python_ver = python_ver.split()[0]  # Get just version number
        st.metric("Python Version", python_ver)
    
    with col2:
        platform = latest.get("platform", "N/A")
        if platform != "N/A":
            platform = platform.split("-")[0]  # Simplify platform string
        st.metric("Platform", platform)
    
    with col3:
        hostname = latest.get("hostname", "N/A")
        st.metric("Hostname", hostname)
    
    with col4:
        cuda_info = latest.get("cuda", {})
        cuda_available = "Yes" if cuda_info.get("nvidia_smi") else "No"
        st.metric("CUDA Available", cuda_available)
    
    # Environment variables
    st.markdown("---")
    st.markdown("### 🔧 Environment Variables")
    
    env_vars = latest.get("env_vars", {})
    
    if env_vars:
        env_col1, env_col2 = st.columns(2)
        
        with env_col1:
            st.markdown("**Reproducibility Settings**")
            repro_vars = {
                k: v for k, v in env_vars.items()
                if k in ["PYTHONHASHSEED", "CUDNN_DETERMINISTIC", "OMP_NUM_THREADS"]
            }
            if repro_vars:
                st.json(repro_vars)
            else:
                st.info("No reproducibility env vars set")
        
        with env_col2:
            st.markdown("**GPU Settings**")
            gpu_vars = {
                k: v for k, v in env_vars.items()
                if k in ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"]
            }
            if gpu_vars:
                st.json(gpu_vars)
            else:
                st.info("No GPU env vars set")
    else:
        st.info("No environment variables captured")
    
    # CUDA/GPU Information
    if cuda_info and cuda_info.get("nvidia_smi"):
        st.markdown("---")
        st.markdown("### 🎮 GPU Information")
        
        nvidia_output = cuda_info.get("nvidia_smi", "")
        
        st.code(nvidia_output, language="text")
        
        nvcc_output = cuda_info.get("nvcc", "")
        if nvcc_output:
            with st.expander("NVCC Version Info"):
                st.code(nvcc_output, language="text")
    
    # PyTorch state
    torch_state = latest.get("torch_state")
    if torch_state:
        st.markdown("---")
        st.markdown("### 🔥 PyTorch Information")
        
        torch_col1, torch_col2, torch_col3 = st.columns(3)
        
        with torch_col1:
            st.metric(
                "PyTorch Version",
                torch_state.get("torch_version", "N/A")
            )
        
        with torch_col2:
            cuda_available = "✅ Yes" if torch_state.get("cuda_available") else "❌ No"
            st.metric("CUDA Available", cuda_available)
        
        with torch_col3:
            cuda_ver = torch_state.get("cuda_version", "N/A")
            st.metric("CUDA Version", cuda_ver if cuda_ver else "N/A")
    
    # Historical comparison
    st.markdown("---")
    st.markdown("### 📈 Historical Systems")
    
    if len(experiments_sys_info) > 1:
        # Show platform consistency
        platforms = [exp.get("platform", "Unknown") for exp in experiments_sys_info]
        unique_platforms = len(set(platforms))
        
        if unique_platforms == 1:
            st.success(f"✅ All {len(experiments_sys_info)} experiments ran on consistent platform: {platforms[0]}")
        else:
            st.warning(f"⚠️ Experiments ran on {unique_platforms} different platforms")
            
            # Show breakdown
            platform_counts = {}
            for p in platforms:
                platform_counts[p] = platform_counts.get(p, 0) + 1
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(platform_counts.keys()),
                    values=list(platform_counts.values()),
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title="Platform Distribution",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Python version consistency
        python_versions = [exp.get("python_version", "Unknown").split()[0] for exp in experiments_sys_info]
        unique_python = len(set(python_versions))
        
        if unique_python == 1:
            st.success(f"✅ Consistent Python version: {python_versions[0]}")
        else:
            st.warning(f"⚠️ {unique_python} different Python versions detected")
    
    # Dependencies comparison
    st.markdown("---")
    st.markdown("### 📦 Dependencies")
    
    if latest.get("pip_freeze"):
        with st.expander("View pip freeze output"):
            st.code(latest["pip_freeze"], language="text")
    
    # Conda info
    conda_info = latest.get("conda", {})
    if conda_info and conda_info.get("info"):
        with st.expander("View conda environment"):
            st.json(conda_info.get("info"))
    
    # Reproducibility score
    st.markdown("---")
    st.markdown("### 🎯 Reproducibility Score")
    
    score = 0
    max_score = 5
    feedback = []
    
    # Check criteria
    if env_vars.get("PYTHONHASHSEED"):
        score += 1
        feedback.append("✅ PYTHONHASHSEED is set")
    else:
        feedback.append("❌ PYTHONHASHSEED not set")
    
    if latest.get("pip_freeze"):
        score += 1
        feedback.append("✅ Dependencies captured (pip freeze)")
    else:
        feedback.append("❌ Dependencies not captured")
    
    if cuda_info:
        score += 1
        feedback.append("✅ CUDA info captured")
    else:
        feedback.append("⚠️ CUDA info not available")
    
    if torch_state:
        score += 1
        feedback.append("✅ PyTorch state captured")
    else:
        feedback.append("❌ PyTorch state not captured")
    
    git_info = latest.get("git", {})
    if git_info and git_info.get("commit"):
        score += 1
        feedback.append("✅ Git commit captured")
    else:
        feedback.append("❌ Git info not captured")
    
    # Display score
    score_pct = (score / max_score) * 100
    
    if score_pct >= 80:
        color = "green"
        emoji = "🎉"
        message = "Excellent"
    elif score_pct >= 60:
        color = "orange"
        emoji = "👍"
        message = "Good"
    else:
        color = "red"
        emoji = "⚠️"
        message = "Needs Improvement"
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea22 0%, #764ba244 100%); 
                padding: 2rem; border-radius: 12px; text-align: center;'>
        <h2 style='margin: 0; font-size: 3rem;'>{emoji}</h2>
        <h3 style='margin: 0.5rem 0; color: {color};'>{score}/{max_score} - {message}</h3>
        <p style='margin: 0; color: #666;'>{score_pct:.0f}% Reproducibility</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    for item in feedback:
        st.markdown(item)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_system_monitor(Path("artifacts"))
