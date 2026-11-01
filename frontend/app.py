
import streamlit as st
from member import show_member_portal

st.set_page_config(
    page_title="CareAgents",
    page_icon="🏥",
    layout="wide",
)

# Initialize the development session.
if "role" not in st.session_state:
    st.session_state.role = None

if st.session_state.role is None:
    st.title("🏥 CareAgents")
    st.subheader("Intelligent Healthcare Assistance")
    st.write("Select a role to explore the application.")

    role = st.selectbox(
        "Continue as",
        ["Member", "Staff", "Admin"],
    )

    if st.button("Continue", type="primary"):
        st.session_state.role = role
        st.rerun()

else:
    role = st.session_state.role

    with st.sidebar:
        st.title("🏥 CareAgents")
        st.write(f"Current role: {role}")

        if st.button("Switch Role"):
            st.session_state.role = None
            st.rerun()

    if role == "Member":
        if role == "Member":
            show_member_portal()

    elif role == "Staff":
        st.title("Staff Portal")
        st.write("Manage cases requiring human review.")
        st.info("The review queue will appear here.")

    elif role == "Admin":
        st.title("Admin Portal")
        st.write("Monitor and manage the CareAgents system.")
        st.info("System status, ingestion, and graph will appear here.")
