import os
import streamlit as st
from rag_engine import LocalVaultRAG

# Page configuration
st.set_page_config(
    page_title="LocalVault AI — Assistant Documentaire 100% Local",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00539b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        background-color: #ecfdf5;
        color: #047857;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #a7f3d0;
    }
    .source-box {
        background-color: #f8fafc;
        border-left: 3px solid #00539b;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.9rem;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG Engine in session state
if "rag" not in st.session_state:
    st.session_state.rag = LocalVaultRAG()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Hardware, Model & Document Settings
with st.sidebar:
    st.image("https://img.shields.io/badge/ASUS%20Agent-Computer%20Challenge-00539b?style=for-the-badge", use_container_width=True)
    st.title("⚙️ Paramètres de l'Agent")
    
    st.markdown("### 🖥️ Modèle LLM Local")
    model_choice = st.selectbox(
        "Sélectionnez le modèle local :",
        ["llama3:latest", "mistral:latest", "phi3:mini", "gemma2:9b"],
        index=0
    )
    st.session_state.rag.set_model(model_choice)
    
    st.markdown("---")
    st.markdown("### 📂 Ingestion de Documents")
    uploaded_files = st.file_uploader(
        "Déposez vos documents (PDF, TXT, MD) :",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("📥 Indexer les fichiers", use_container_width=True):
            with st.spinner("Indexation vectorielle en cours..."):
                count = st.session_state.rag.ingest_files(uploaded_files)
                st.success(f"✓ {count} document(s) indexé(s) dans ChromaDB !")
                
    st.markdown("---")
    st.markdown("### 🔒 Statut de Confidentialité")
    st.markdown('<span class="status-badge">✓ 100% Hors-Ligne (Zero-Cloud)</span>', unsafe_allow_html=True)
    st.caption("Aucune donnée ne quitte votre machine. Inférence sur GPU local.")

# Main Interface Header
col_header, col_badge = st.columns([3, 1])
with col_header:
    st.markdown('<div class="main-title">🔒 LocalVault AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interrogez vos cours, factures et documents en toute confidentialité sur votre PC.</div>', unsafe_allow_html=True)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📌 Sources consultées"):
                for src in message["sources"]:
                    st.markdown(f'<div class="source-box"><strong>Fichier :</strong> {src.get("file", "Inconnu")} (Extrait : <em>{src.get("snippet", "")}</em>)</div>', unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Posez une question sur vos documents locaux..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Recherche locale et réflexion en cours..."):
            response_text, sources = st.session_state.rag.query(prompt)
            st.markdown(response_text)
            
            if sources:
                with st.expander("📌 Sources consultées"):
                    for src in sources:
                        st.markdown(f'<div class="source-box"><strong>Fichier :</strong> {src.get("file", "Inconnu")} (Extrait : <em>{src.get("snippet", "")}</em>)</div>', unsafe_allow_html=True)
                        
    # Save assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources
    })
