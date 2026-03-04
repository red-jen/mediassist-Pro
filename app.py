import streamlit as st
import os
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

def register_user(username: str, email: str, password: str) -> dict:
    """Register a new user via the backend API."""
    response = requests.post(
        f"{API_URL}/auth/register",
        json={"username": username, "email": email, "password": password}
    )
    return response.json() if response.status_code == 200 else {"error": response.json().get("detail", "Registration failed")}

def login_user(username: str, password: str) -> dict:
    """Login and get JWT token."""
    response = requests.post(
        f"{API_URL}/auth/login",
        data={"username": username, "password": password}                         
    )
    if response.status_code == 200:
        return response.json()
    return {"error": response.json().get("detail", "Login failed")}

def ask_question(query: str, chat_history: list, token: str) -> dict:
    """Send a question to the RAG backend."""
    response = requests.post(
        f"{API_URL}/rag/ask",
        json={"query": query, "chat_history": chat_history},
        headers={"Authorization": f"Bearer {token}"}
    )
    if response.status_code == 200:
        return response.json()
    return {"error": response.json().get("detail", "Query failed")}

def upload_pdf(file_bytes: bytes, filename: str, token: str) -> dict:
    """Upload a PDF to be indexed."""
    files = {"file": (filename, file_bytes, "application/pdf")}
    response = requests.post(
        f"{API_URL}/rag/upload-pdf",
        files=files,
        headers={"Authorization": f"Bearer {token}"}
    )
    if response.status_code == 200:
        return response.json()
    return {"error": response.json().get("detail", "Upload failed")}

st.set_page_config(
    page_title="MediAssist Pro",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 MediAssist Pro")
st.caption("Posez vos questions sur vos manuels de dispositifs médicaux.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_pdfs" not in st.session_state:
    st.session_state.indexed_pdfs = []                                       

if "token" not in st.session_state:
    st.session_state.token = None                               

if "username" not in st.session_state:
    st.session_state.username = None

with st.sidebar:

    if st.session_state.token is None:
        st.header("🔐 Authentication")

        auth_tab = st.radio("", ["Login", "Register"], horizontal=True, label_visibility="collapsed")

        if auth_tab == "Login":
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")

                if submitted and username and password:
                    result = login_user(username, password)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.session_state.token = result["access_token"]
                        st.session_state.username = username
                        st.success("Logged in!")
                        st.rerun()

        else:            
            with st.form("register_form"):
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Register")

                if submitted and username and email and password:
                    result = register_user(username, email, password)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("Registered! Please login.")

    else:

        st.success(f"👤 Logged in as **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.header("📄 Documents")
        uploaded_file = st.file_uploader(
            "Upload a PDF manual",
            type=["pdf"],
            help="You can upload multiple PDFs one by one"
        )

        if uploaded_file:
            if uploaded_file.name not in st.session_state.indexed_pdfs:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    file_bytes = uploaded_file.read()
                    result = upload_pdf(file_bytes, uploaded_file.name, st.session_state.token)

                if "error" in result:
                    st.error(result["error"])
                elif result.get("status") == "indexed":
                    st.session_state.indexed_pdfs.append(uploaded_file.name)
                    st.success(f"Indexed {result.get('chunks', '?')} chunks from **{result.get('source', uploaded_file.name)}**")
                else:
                    st.info(f"**{result.get('source', uploaded_file.name)}** already indexed")
            else:
                st.info(f"**{uploaded_file.name}** already indexed")

        if st.session_state.indexed_pdfs:
            st.markdown("**Indexed documents:**")
            for name in st.session_state.indexed_pdfs:
                st.write(f"✅ {name}")

        st.divider()

        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question..."):

    if st.session_state.token is None:
        st.warning("Please login first.")
        st.stop()

    if not st.session_state.indexed_pdfs:
        st.warning("Please upload at least one PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question(prompt, st.session_state.chat_history, st.session_state.token)

        if "error" in result:
            st.error(result["error"])
            answer = "Sorry, an error occurred."
            sources = []
        else:
            answer = result.get("answer", "No answer received.")
            sources = result.get("sources", [])

        st.markdown(answer)

        if sources:
            with st.expander("📚 Sources"):
                for s in sources:
                    st.write(f"- **{s.get('source', 'Unknown')}** — page {s.get('page', '?')}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append({"question": prompt, "answer": answer})
