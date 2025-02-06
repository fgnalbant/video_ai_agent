import streamlit as st
import time
import tempfile
from pathlib import Path
from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Video  # Video nesnesi için
from agno.tools.duckduckgo import DuckDuckGoTools

st.set_page_config(
    page_title="Multimodal AI Agent",
    page_icon="",
    layout="wide"
)

st.title("Multimodal AI Agent ")

# Initialize the agent with Gemini model
@st.cache_resource
def initialize_agent():
    return Agent(
        name="Multimodal Analyst",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGoTools()],
        markdown=True,
    )

agent = initialize_agent()

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    # Kaydedilen dosyayı geçici bir dosyaya yazın
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    st.video(video_path)
    
    user_prompt = st.text_area(
        "What would you like to know?",
        placeholder="Ask any question related to the video - the AI Agent will analyze it and search the web if needed",
        help="You can ask questions about the video content and get relevant information from the web"
    )
    
    if st.button("Analyze & Research"):
        if not user_prompt:
            st.warning("Please enter your question.")
        else:
            with st.spinner("Processing video and researching..."):
                try:
                    # Video nesnesini oluşturuyoruz
                    video_obj = Video(filepath=Path(video_path))
                    
                    # Gemini modelinin video analiz yeteneğini kullanarak direkt video gönderiyoruz
                    prompt = f"""
                    First analyze this video and then answer the following question using both 
                    the video analysis and web research: {user_prompt}
                    
                    Provide a comprehensive response focusing on practical, actionable information.
                    """
                    
                    # agent.run veya agent.print_response kullanılabilir.
                    # Örneğin agent.print_response kullanımı:
                    response = agent.run(prompt, videos=[video_obj])
                    
                    st.subheader("Result")
                    st.markdown(response.content)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    Path(video_path).unlink(missing_ok=True)
else:
    st.info("Please upload a video to begin analysis.")

# CSS Styling for Text Area
st.markdown("""
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """, unsafe_allow_html=True)