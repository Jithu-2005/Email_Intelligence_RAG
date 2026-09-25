import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"   

st.title("MailMind — Email Intelligence")
st.caption("Ask questions about your emails. Every answer cites the emails it came from.")

with st.sidebar:
    if st.button("Sync mailbox (sample data)"):
        with st.spinner("Indexing emails..."):
            r = requests.post(f"{API_BASE}/ingest", json={"source": "sample"}, timeout=180)
        st.success(f"Indexed {r.json().get('chunks_indexed', 0)} chunks")

    st.subheader("Try asking:")
    st.write("- Draft a reply to Rahul confirming the 23 Sept meeting")        
    st.write("- Which attachments came with the vendor invoice?")               
    st.write("- What is the invoice number and amount due?")                    
    st.write("- Find emails from Rahul after 15 Sept about the proposal")       
    st.write("- Summarize the client sync thread")                             

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input("Ask about your emails..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    response = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=90)
    if response.ok:
        data = response.json()
        tools_used = ", ".join(data["tools_used"]) or "no tool (general reply)"
        sources = ", ".join(data["sources"]) or "none"
        full = f"{data['answer']}\n\n_Tools used: {tools_used}_  \n_Emails cited: {sources}_"
    else:
        full = f"Something went wrong: {response.text}"

    st.session_state.messages.append({"role": "assistant", "content": full})
    st.chat_message("assistant").write(full)
