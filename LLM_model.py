import streamlit as st
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Initialize model and processor
from transformers import MllamaForConditionalGeneration

model_id = "meta-llama/Llama-3.2-90B-Vision"
access_token = "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"  # replace with your token

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    use_auth_token=access_token,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_id)

# Streamlit app
st.title("Simple OCR App")
st.write("Upload an image, and the app will extract text present in the image.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Use OCR prompt
    prompt = "<|image|><|begin_of_text|>Extract text visible on the image: "
    inputs = processor(image, prompt, return_tensors="pt").to(model.device)

    # Generate output without adding extra information
    output = model.generate(**inputs, max_new_tokens=30)
    extracted_text = processor.decode(output[0]).strip()

    # Display the extracted text
    st.write("**Extracted Text:**")
    st.write(extracted_text)
