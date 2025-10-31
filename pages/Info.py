"""Info page displaying explanatory content about models, seeds, and metrics."""

from pathlib import Path

import streamlit as st

# Define path relative to this file's location
PAGE_ROOT = Path(__file__).resolve().parent.parent
EXPLAINER_PATH = PAGE_ROOT / "docs" / "explainers.md"


@st.cache_data(show_spinner=False)
def load_explainers() -> str | None:
    """Load explainer content from markdown file."""
    if not EXPLAINER_PATH.exists():
        return None
    return EXPLAINER_PATH.read_text(encoding="utf-8")


st.set_page_config(
    page_title="Info - Synthetic Data Workbench", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📚"
)

# Hide Streamlit's automatic page navigation using CSS and JavaScript
# Also add styling to highlight the current page
nav_css = """
<style>
/* Hide Streamlit's automatic navigation completely - multiple selectors for reliability */
[data-testid="stSidebarNav"],
[data-testid="stSidebarNav"] *,
nav[data-testid="stSidebarNav"],
nav[data-testid="stSidebarNav"] ul,
nav[data-testid="stSidebarNav"] li,
section[data-testid="stSidebarNav"],
section[data-testid="stSidebarNav"] > * {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
}
/* Highlight the Info page link */
div[data-testid="stSidebar"] a[href*="Info.py"] {
    background-color: rgba(38, 39, 48, 0.6);
    border-left: 3px solid #ff6b6b;
    padding-left: 1rem;
    border-radius: 0.25rem;
}
</style>
<script>
// Ensure navigation is hidden even if CSS doesn't catch it initially
window.addEventListener('load', function() {
    const nav = document.querySelector('[data-testid="stSidebarNav"]');
    if (nav) {
        nav.style.display = 'none';
        nav.style.visibility = 'hidden';
        nav.style.height = '0';
        nav.style.width = '0';
    }
});
</script>
"""
st.markdown(nav_css, unsafe_allow_html=True)

# Add custom navigation links in sidebar
st.sidebar.markdown("### Navigation")
st.sidebar.page_link("streamlit_app.py", label="🏠 Home")
st.sidebar.page_link("pages/Info.py", label="📚 Info & Documentation")
st.sidebar.markdown("---")

st.title("Info & Documentation")

explainer_md = load_explainers()
if explainer_md:
    st.markdown(explainer_md)
else:
    st.info("No additional information found. Please check `docs/explainers.md`.")
    st.markdown("""
    ## About This Application
    
    This Synthetic Data Workbench helps you generate synthetic tabular data using SDV (Synthetic Data Vault).
    
    ### Key Concepts
    
    - **Models**: Choose from CTGAN, GaussianCopula, or TVAE synthesizers
    - **Seeds**: Use random seeds for reproducible results
    - **Utility Score**: Measures how similar synthetic data is to original (0-1 scale)
    - **Metadata**: Automatically detected from your CSV structure
    """)

