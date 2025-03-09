import streamlit as st
import os
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding

# Load environment variables
load_dotenv()
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY", "your_default_api_key")

# Enable asynchronous tasks
nest_asyncio.apply()

# Configure LLM Settings
Settings.llm = MistralAI(api_key=os.environ["MISTRAL_API_KEY"])
Settings.embed_model = MistralAIEmbedding(api_key=os.environ["MISTRAL_API_KEY"])

# Define policies with descriptions
POLICIES = {
    "Student Conduct Policy": "This policy governs student behavior and disciplinary actions...",
    "Academic Schedule Policy": "This policy outlines the academic calendar and scheduling rules...",
    "Student Attendance Policy": "This policy explains attendance requirements and consequences of absenteeism...",
    "Student Appeals Policy": "This policy details the process for students to appeal decisions...",
    "Graduation Policy": "This policy describes graduation requirements and processes...",
    "Academic Standing Policy": "This policy defines academic performance standards...",
    "Transfer Policy": "This policy outlines the transfer of credits and student mobility...",
    "Admissions Policy": "This policy sets the criteria and procedures for student admissions...",
    "Final Grade Policy": "This policy explains the grading system and final grade calculations...",
    "Registration Policy": "This policy provides guidelines for course registration and enrollment...",
}

# Generate policy URLs dynamically
POLICY_URLS = {
    name: f"https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/{name.lower().replace(' ', '-')}"
    for name in POLICIES.keys()
}

# Create document index for policies
documents = [Document(text=POLICIES[name], metadata={"name": name}) for name in POLICIES]
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Streamlit UI Design
st.set_page_config(page_title="UDST Policy Assistant", layout="wide")
st.markdown("""
    <style>
        .main-title {text-align: center; font-size: 32px; font-weight: bold;}
        .sub-text {text-align: center; font-size: 18px; color: gray;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='main-title'>📜 UDST Policy Assistant</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Find answers to your queries on university policies effortlessly.</p>", unsafe_allow_html=True)

# User Input Section
st.write("### 🔍 Ask a Question:")
user_query = st.text_input("Enter your question about UDST policies:")

if user_query:
    query_lower = user_query.lower()
    matched_policies = [name for name in POLICIES.keys() if query_lower in name.lower()]

    if matched_policies:
        st.write("#### 📚 Relevant Policies:")
        for policy in matched_policies:
            st.markdown(f"- [{policy}]({POLICY_URLS[policy]})")

        # Generate response from the query engine using the relevant policies
        relevant_documents = [Document(text=POLICIES[name], metadata={"name": name}) for name in matched_policies]
        relevant_query_engine = VectorStoreIndex.from_documents(relevant_documents).as_query_engine()
        response = relevant_query_engine.query(user_query)
        
        st.write("### 🤖 AI Response:")
        st.success(response.response if response else "No relevant information found.")
    else:
        st.warning("No matching policies found. Please refine your query.")

# Footer
st.markdown("---")
st.markdown("<p class='sub-text'>Powered by AI | UDST Policy Search Assistant</p>", unsafe_allow_html=True)
