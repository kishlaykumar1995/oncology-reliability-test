import streamlit as st
import json
import os
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Configuration & Layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OncoSift Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Cards"
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border: 1px solid #d6d6d6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .Reliability-High {
        border-left: 5px solid #28a745; 
    }
    .Reliability-Medium {
        border-left: 5px solid #ffc107;
    }
    .Reliability-Low {
        border-left: 5px solid #dc3545;
    }
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Data Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    file_path = 'extracted_oncology_data.json'
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

data = load_data()
df = pd.DataFrame(data)

# Helper to safely get nested values (since DataFrame flattens JSON awkwardly sometimes)
def get_nested(record, *keys):
    val = record
    for key in keys:
        if isinstance(val, dict):
            val = val.get(key, {})
        else:
            return None
    return val if not isinstance(val, dict) else None

# -----------------------------------------------------------------------------
# 3. Sidebar - Stats
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧬 OncoSift")
    st.markdown("---")
    
    if not df.empty:
        total_docs = len(df)
        high_rel = len([d for d in data if d.get('predicted_reliability') == 'High'])
        
        col1, col2 = st.columns(2)
        col1.metric("Documents", total_docs)
        col2.metric("High Reliability", high_rel)
        
        st.markdown("### Reliability Distribution")
        reliability_counts = pd.Series([d.get('predicted_reliability', 'Unknown') for d in data]).value_counts()
        st.bar_chart(reliability_counts)
    else:
        st.warning("No data found. Please ensure 'extracted_oncology_data.json' exists.")

# -----------------------------------------------------------------------------
# 4. Main Page - Search & Tabs
# -----------------------------------------------------------------------------
st.markdown("## AI-Assisted Oncology Research Intelligence")

# Search Bar (RAG Placeholder)
search_query = st.text_input("🔍 Ask a question about the research (e.g., 'What is the pCR rate in KEYNOTE-522?')", "")
if search_query:
    st.info(f"RAG Search is under construction. Searching for: **{search_query}**")
    # Here you would call your RAG backend
    
st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Oncology", "Cardiology (Coming Soon)", "Neurology (Coming Soon)"])

with tab1:
    # -------------------------------------------------------------------------
    # 5. Filters (Dropdowns)
    # -------------------------------------------------------------------------
    if not df.empty:
        # Extract unique cancer types for the dropdown
        cancer_types = sorted(list(set([d['population']['cancer_type'] for d in data])))
        cancer_types.insert(0, "All Cancers")
        
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            selected_cancer = st.selectbox("Select Disease Subtype", cancer_types)
        with c2:
            reliability_filter = st.multiselect(
                "Filter by Reliability", 
                ["High", "Medium", "Low"], 
                default=["High", "Medium", "Low"]
            )
        
        # Filter Logic
        filtered_data = [d for d in data if 
                         (selected_cancer == "All Cancers" or d['population']['cancer_type'] == selected_cancer) and
                         (d.get('predicted_reliability') in reliability_filter)
                        ]
        
        st.markdown(f"**Showing {len(filtered_data)} results**")
        st.markdown("---")

        # ---------------------------------------------------------------------
        # 6. Display Cards
        # ---------------------------------------------------------------------
        for doc in filtered_data:
            reliability = doc.get('predicted_reliability', 'Low')
            
            # Create a container with a visual border indicating reliability
            with st.container(border=True):
                # Header Row
                col_head1, col_head2 = st.columns([4, 1])
                with col_head1:
                    st.subheader(doc.get('title', 'Untitled Study'))
                    st.caption(f"📄 {doc.get('filename')} | 🗓️ {doc.get('publication_year')}")
                with col_head2:
                    # Color-coded badge
                    color = "green" if reliability == "High" else "orange" if reliability == "Medium" else "red"
                    st.markdown(f":{color}[**{reliability} Reliability**]")
                    
                # Content Grid
                c_info, c_design, c_results = st.columns(3)
                
                with c_info:
                    st.markdown("**🧑‍🤝‍🧑 Population**")
                    st.write(f"**Type:** {doc['population'].get('cancer_type')}")
                    st.write(f"**N:** {doc['design'].get('sample_size')}")
                    st.write(f"**Age:** {doc['population'].get('patient_age_range')}")

                with c_design:
                    st.markdown("**🔬 Design**")
                    st.write(f"**Type:** {doc['design'].get('study_type')}")
                    st.write(f"**Intervention:** {doc['design'].get('intervention')}")
                    st.write(f"**Comparator:** {doc['design'].get('comparator', 'None')}")

                with c_results:
                    st.markdown("**📊 Key Outcome**")
                    st.info(doc['results'].get('primary_result_numeric', 'N/A'))
                    st.write(f"_{doc['results'].get('primary_result_text')}_")

                # Expandable details
                with st.expander("See Detailed Extraction"):
                    st.json(doc)

    else:
        st.info("No documents extracted yet. Please run the extraction pipeline first.")