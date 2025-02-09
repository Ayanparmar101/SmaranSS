import pathlib
import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
import streamlit_antd_components as sac
from utils import get_answer, text_to_speech, autoplay_audio, speech_to_text

# New imports for RAG functionality
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import tempfile

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Social Science Guide", 
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("assets/styles.css")
load_css(css_path)

# Custom background style
hide_st_style = """
<style>
[data-testid="stAppViewContainer"]{ 
background-color: #000000;
opacity: 1;
background: linear-gradient(135deg, #21212155 25%, transparent 25%) -40px 0/ 80px 80px, linear-gradient(225deg, #212121 25%, transparent 25%) -40px 0/ 80px 80px, linear-gradient(315deg, #21212155 25%, transparent 25%) 0px 0/ 80px 80px, linear-gradient(45deg, #212121 25%, #000000 25%) 0px 0/ 80px 80px;
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load environment variables
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found! Please check your .env file")
    st.stop()
client = OpenAI(api_key=api_key)

# Define allowed topics and chapters with PDF paths
allowed_topics = {
    "Democratic Politics I": {
        "chapters": [
            "What Is Democracy? Why Democracy?", 
            "Constitutional Design",
            "Electoral Politics", 
            "Working of Institutions", 
            "Democratic Rights"
        ],
        "pdf_path": {
            "What Is Democracy? Why Democracy?": "pdfs/democratic_politics/ch1.pdf",
            # Add paths for other chapters
        }
    },
    "Contemporary India": {
        "chapters": [
            "India – Size and Location", 
            "Physical Features of India",
            "Drainage", 
            "Climate", 
            "Natural Vegetation and Wildlife", 
            "Population"
        ],
        "pdf_path": {
            "India – Size and Location": "pdfs/contemporary_india/ch1.pdf",
            # Add paths for other chapters
        }
    },
    # Add other topics similarly
}

# Function to load and process PDF for RAG
def get_chapter_context(pdf_path):
    # Load and process PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(pages)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings
    )
    return vectorstore.as_retriever()

# Function to stream responses from OpenAI
def stream_response(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as e:
        yield f"An error occurred: {str(e)}"

# Function to get concept help with RAG
def get_concept_help(topic, chapter, question):
    if topic not in allowed_topics or chapter not in allowed_topics[topic]["chapters"]:
        return "Error: Invalid topic or chapter selection."

    # Get PDF path
    pdf_path = allowed_topics[topic]["pdf_path"].get(chapter)
    if not pdf_path or not os.path.exists(pdf_path):
        st.error("PDF resource not found for this chapter!")
        st.stop()

    # Retrieve relevant context
    retriever = get_chapter_context(pdf_path)
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Updated system message
    system_msg = f"""You are a CBSE Class 9 Social Science tutor. Use this context from the NCERT textbook:
    {context}
    
    - Provide explanations strictly from the given material
    - Highlight relevant sections from the textbook
    - Maintain original NCERT terminology and structure"""

    user_prompt = f"""Help the student understand {chapter} from {topic} (Class 9 NCERT). Their question: {question}
    Provide:
    1. Relevant NCERT chapter reference
    2. Key concepts with clear explanation
    3. Real-world examples (if applicable)
    4. Previous years' CBSE question patterns"""

    return stream_response([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt}
    ])

# Function to analyze answers
def analyze_answer(text):
    system_msg = """You are a CBSE Social Science answer evaluator.
    - Check alignment with NCERT content
    - Evaluate using CBSE marking scheme format
    - Identify missing key points
    - Suggest improvements
    - Focus on exam-oriented writing"""

    return stream_response([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Evaluate this Social Science answer:\n{text}"}
    ])

# Function to get direct answers
def get_direct_answer(topic, chapter):
    if topic not in allowed_topics or chapter not in allowed_topics[topic]["chapters"]:
        def error_generator():
            yield "Error: Invalid topic or chapter selection."
        return error_generator()

    system_msg = """You are a CBSE Class 9 Social Science expert.
    - Provide model answers strictly from NCERT textbook content.
    - Follow CBSE format, including introduction, body, and conclusion.
    - Use NCERT-approved terminology and structured explanations.
    - Include dates, facts, and relevant diagrams/maps where applicable."""

    user_prompt = f"Provide a Class 9 NCERT-based model answer for {chapter} from {topic}"

    return stream_response([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt}
    ])

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = '📚 Concept Helper'
if 'concept_initial' not in st.session_state:
    st.session_state.concept_initial = False

# Sidebar navigation
with st.sidebar:
    pages = ["📚 Concept Helper", "✍️ Answer Evaluation", "🗣️ Interactive Tutor"]
    selected_index = sac.segmented(
        items=[
            sac.SegmentedItem(label='📚 Concept Helper'),
            sac.SegmentedItem(label='✍️ Answer Evaluation'),
            sac.SegmentedItem(label='🗣️ Interactive Tutor'),
        ],
        label='',
        index=pages.index(st.session_state.page),
        align='center',
        direction='vertical',
        size='xl',
        radius='xs',
        color='rgb(20,80,90)',
        divider=False,
        key='nav_segmented'
    )
    st.session_state.page = selected_index

# Concept Helper Page
if st.session_state.page == "📚 Concept Helper":
    st.markdown("<h1 class='comic-title'>📚 Social Science Concept Helper</h1>", unsafe_allow_html=True)
    st.markdown("""<div class="custom-markdown">Your personal CBSE Class 9 Social Science guide!</div>""", unsafe_allow_html=True)
    st.markdown("---")

    topic = st.selectbox("Select Subject", options=list(allowed_topics.keys()))
    chapter = st.selectbox("Select Chapter", options=allowed_topics[topic]["chapters"])
    prompt = st.chat_input("Ask your question about this chapter...")
    
    if prompt:
        if prompt.strip() == "":
            st.warning("Please enter your question first!")
        else:
            st.subheader("Guidance:")
            response = get_concept_help(topic, chapter, prompt)
            st.write_stream(response)
            st.session_state.concept_initial = True
            st.session_state.concept_topic = topic
            st.session_state.concept_chapter = chapter
            st.markdown("---")
            st.success("Try solving related questions from NCERT exercises!")

    if st.session_state.concept_initial and st.session_state.page == '📚 Concept Helper':
        if st.button("Show Model Answer"):
            st.subheader("CBSE Format Answer:")
            direct_response = get_direct_answer(st.session_state.concept_topic, st.session_state.concept_chapter)
            st.write_stream(direct_response)

# Answer Evaluation Page
elif st.session_state.page == "✍️ Answer Evaluation":
    st.markdown("<h1 class='comic-title'>✍️ Answer Evaluation</h1>", unsafe_allow_html=True)
    st.markdown("""<div class="custom-markdown">Get your answers evaluated as per CBSE guidelines</div>""", unsafe_allow_html=True)
    
    prompt = st.chat_input("Paste your answer here...")
    if prompt:
        if prompt.strip() == "":
            st.warning("Please enter your answer for evaluation!")
        else:
            st.subheader("Evaluation Feedback")
            analysis = analyze_answer(prompt)
            st.write_stream(analysis)
            st.success("Review the feedback and revise your answer!")

# Interactive Tutor Page
elif st.session_state.page == "🗣️ Interactive Tutor":
    st.markdown("<h1 class='comic-title'>Interactive Social Science Tutor 🌍</h1>", unsafe_allow_html=True)
    float_init()

    if "socratic_messages" not in st.session_state:
        st.session_state.socratic_messages = [
            {"role": "system", "content": """You are a CBSE Class 9 Social Science tutor. 
            Guide students through concepts using the Socratic method and NCERT curriculum."""},
            {"role": "assistant", "content": "Which Social Science topic would you like to explore today?"}
        ]

    footer_container = st.container()
    with footer_container:
        audio_bytes = audio_recorder()

    for message in st.session_state.socratic_messages:
        if message["role"] == "system": continue
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Type your question here...")
    if prompt:
        st.session_state.socratic_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if audio_bytes:
        with st.spinner("Transcribing..."):
            webm_file_path = "temp_audio.webm"
            with open(webm_file_path, "wb") as f:
                f.write(audio_bytes)
            
            try:
                transcript = speech_to_text(webm_file_path)
                if transcript:
                    st.session_state.socratic_messages.append({"role": "user", "content": transcript})
                    with st.chat_message("user"):
                        st.write(transcript)
                else:
                    st.error("Transcription failed. Please try again.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if os.path.exists(webm_file_path):
                    os.remove(webm_file_path)

    if st.session_state.socratic_messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking🤔..."):
                final_response = get_answer([msg for msg in st.session_state.socratic_messages if msg["role"] != "system"])
            with st.spinner("Generating audio response..."):
                audio_file = text_to_speech(final_response)
                autoplay_audio(audio_file)
            st.write(final_response)
            st.session_state.socratic_messages.append({"role": "assistant", "content": final_response})
            if os.path.exists(audio_file):
                os.remove(audio_file)

    footer_container.float("bottom: 0rem;")

# Footer
st.markdown("---")
st.caption("Powered by Brain | Made By Ayan Parmar")