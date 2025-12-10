import streamlit as st
import json
import os
import pandas as pd
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message

# -----------------------------------------------------------------------------
# 1. Configuration & Layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OncoSift Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Cards" and citations
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
        color: #31333F; /* Force dark text color */
    }
    .citation-box {
        font-size: 0.85em;
        color: #555;
        background-color: #fff;
        border: 1px solid #eee;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Data Loading & Helper Functions
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

# Helper to safely get nested values
def get_nested(record, *keys):
    val = record
    for key in keys:
        if isinstance(val, dict):
            val = val.get(key, {})
        else:
            return None
    return val if not isinstance(val, dict) else None

# -----------------------------------------------------------------------------
# 3. Pinecone Initialization
# -----------------------------------------------------------------------------
@st.cache_resource
def get_assistant():
    api_key = os.environ.get('PINECONE_API_KEY')
    if not api_key:
        return None
    try:
        pc = Pinecone(api_key=api_key)
        # Initialize the assistant
        assistant = pc.assistant.Assistant(
            assistant_name="oncology-reliability", 
        )
        return assistant
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {e}")
        return None

assistant = get_assistant()

# -----------------------------------------------------------------------------
# 4. Sidebar - Stats
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧬 OncoSift")
    st.markdown("---")
    
    # API Key Check
    if not os.environ.get('PINECONE_API_KEY'):
        st.warning("⚠️ PINECONE_API_KEY not found in environment variables. Search will not work.")
    
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
# 5. Main Page - Search & Tabs
# -----------------------------------------------------------------------------
st.markdown("## AI-Assisted Oncology Research Intelligence")

# Search Bar (Pinecone Assistant Integration)
search_query = st.text_input("🔍 Ask a question about the research (e.g., 'What is the pCR rate in KEYNOTE-522?')", "")

if search_query:
    if not assistant:
        st.error("Pinecone Assistant is not initialized. Check your API Key.")
    else:
        with st.spinner("Consulting oncology knowledge base..."):
            try:
                # Construct message for Pinecone
                msg = Message(role="user", content=search_query)
                
                # Call Pinecone Assistant
                resp = assistant.chat(messages=[msg])
                
                # Depending on SDK version, resp might be an object or dict. 
                # We handle both access patterns below.
                if isinstance(resp, dict):
                    content = resp.get('message', {}).get('content', '')
                    citations = resp.get('citations', [])
                else:
                    # Assuming Pydantic model access
                    content = resp.message.content
                    citations = resp.citations if hasattr(resp, 'citations') else []

                # Display Answer
                st.markdown("### 🤖 Answer")
                st.info(content)

                # Display Citations if available
                if citations:
                    st.markdown("#### 📚 Sources")
                    
                    # Extract unique files from citations structure
                    unique_files = {}
                    # Traverse: citations -> references -> file
                    citation_list = citations if isinstance(citations, list) else [citations]
                    
                    for cit in citation_list:
                        # Handle both object/dict access for nested structures
                        refs = cit.get('references', []) if isinstance(cit, dict) else getattr(cit, 'references', [])
                        
                        for ref in refs:
                            file_obj = ref.get('file', {}) if isinstance(ref, dict) else getattr(ref, 'file', {})
                            
                            # Safely extract fields
                            fname = file_obj.get('name') if isinstance(file_obj, dict) else getattr(file_obj, 'name', 'Unknown File')
                            furl = file_obj.get('signed_url') if isinstance(file_obj, dict) else getattr(file_obj, 'signed_url', '#')
                            
                            if fname and fname not in unique_files:
                                unique_files[fname] = furl

                    # Render sources
                    cols = st.columns(len(unique_files)) if unique_files else [st.container()]
                    for idx, (fname, url) in enumerate(unique_files.items()):
                        with st.expander(f"📄 Source: {fname}"):
                            st.markdown(f"[Download / View PDF]({url})")

            except Exception as e:
                st.error(f"Error retrieving answer: {e}")

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Oncology", "Cardiology (Coming Soon)", "Neurology (Coming Soon)"])

with tab1:
    # -------------------------------------------------------------------------
    # 6. Filters (Dropdowns)
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
        # 7. Display Cards
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