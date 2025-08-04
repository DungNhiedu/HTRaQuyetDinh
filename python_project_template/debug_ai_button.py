"""
Debug script to test the Upload CSV and AI prediction functionality
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

st.set_page_config(page_title="Debug Test", layout="wide")

st.title("🔍 Debug Upload CSV & AI Prediction")

# Show current session state
st.sidebar.markdown("### 📊 Session State Debug")
if st.sidebar.button("Show Session State"):
    st.sidebar.json(dict(st.session_state))

# Simulate the logic from the main app
demo_option = st.selectbox("Demo Option", ["Demo Dữ Liệu Mẫu", "Tải File CSV", "Demo Dự Báo"])

st.markdown(f"**Current demo_option:** {demo_option}")

# Check the AI prediction logic
has_sample_data = demo_option == "Demo Dữ Liệu Mẫu"
has_uploaded_data = (demo_option == "Tải File CSV" and 
                    st.session_state.get('upload_processed', False) and 
                    'uploaded_data' in st.session_state)

st.markdown(f"**has_sample_data:** {has_sample_data}")
st.markdown(f"**has_uploaded_data:** {has_uploaded_data}")
st.markdown(f"**upload_processed:** {st.session_state.get('upload_processed', False)}")
st.markdown(f"**uploaded_data in session:** {'uploaded_data' in st.session_state}")

if has_sample_data or has_uploaded_data:
    st.success("✅ AI Prediction button should be ENABLED")
    if st.button("🧠 Test AI Prediction Button"):
        st.success("AI Prediction button clicked!")
else:
    st.error("❌ AI Prediction button should be DISABLED")
    st.info("💡 Load data to use AI prediction")

# Test upload functionality
if demo_option == "Tải File CSV":
    st.markdown("## Upload Test")
    
    uploaded_files = st.file_uploader("Upload CSV", type=['csv'], accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files")
        
        if st.button("🔄 Test Process Data"):
            # Simulate the processing
            st.session_state['upload_processed'] = True
            st.session_state['uploaded_data'] = pd.DataFrame({'test': [1, 2, 3]})
            st.success("✅ Processing complete - session state updated")
            st.rerun()
