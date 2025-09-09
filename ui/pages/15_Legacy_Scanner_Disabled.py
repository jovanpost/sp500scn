import streamlit as st

def page():
    st.header("Legacy Scanner (temporarily disabled)")
    st.info(
        "The old scanner is parked while we stabilize the new Gap/Volume scanner. "
        "This avoids an import chain that pulled in a removed module (`utils.prices`). "
        "You can use the new scanner and the History & Outcomes pages meanwhile."
    )
    st.caption("If you need something specific from the old scanner, say the word and we’ll stub it here safely.")

# Streamlit multipage expects the module-level call:
if __name__ == "__main__":
    page()
