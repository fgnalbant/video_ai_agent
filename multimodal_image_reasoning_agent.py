import streamlit as st
from agno.agent import Agent
from agno.media import Image as AgnoImage
from agno.models.google import Gemini
import google.generativeai as genai
import tempfile
from dotenv import load_dotenv
import os

# API Key Tanımlama
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(
    page_title="Multimodal AI Agent",
    page_icon="🤖",
    layout="wide"
)

def main():
    # Agent oluştur
    agent = Agent(
        model=Gemini(id="gemini-1.5-flash"),
        markdown=True
    )

    st.title("Multimodal Reasoning AI Agent 🎨")

    st.write(
        "Upload an image and provide a reasoning-based task for the AI Agent. "
        "The AI Agent will analyze the image and respond based on your input."
    )

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Görüntüyü göster
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

            # Kullanıcıdan görev girdisi al
            task_input = st.text_area("Enter your task/question for the AI Agent:")

            # İşlem başlatma butonu
            if st.button("Analyze Image") and task_input:
                with st.spinner("AI is thinking..."):
                    try:
                        # Geçici dosya oluştur ve resmi kaydet
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                            temp_file.write(uploaded_file.read())
                            temp_filepath = temp_file.name

                        # Yeni AgnoImage formatına uygun nesne oluştur
                        agno_image = AgnoImage(filepath=temp_filepath)

                        # Agent çalıştır
                        response = agent.run(task_input, images=[agno_image])

                        # Sonucu göster
                        st.markdown("### AI Response:")
                        st.write(response.content)
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
