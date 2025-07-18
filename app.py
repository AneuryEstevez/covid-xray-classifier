import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="COVID X-ray Classifier",
    page_icon="🔬",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("covid_xray_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Preprocess image function
def preprocess_image(image):
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 224x224 as expected by the model
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    image_array = np.array(image)
    image_array = image_array.astype('float32')
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Main app
def main():
    st.title("COVID X-ray Classification")
    st.markdown("Upload an X-ray image to classify it as COVID-19, Non-COVID, or Normal")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please ensure 'covid_xray_model.keras' is in the same directory.")
        return
    
    # Class names
    class_names = ['COVID-19', 'Non-COVID', 'Normal']
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_container_width=True)
            
            # Add predict button
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    predictions = model.predict(processed_image)
                    predicted_class_index = np.argmax(predictions[0])
                    predicted_class = class_names[predicted_class_index]
                    confidence = predictions[0][predicted_class_index]
                    
                    # Store results in session state
                    st.session_state['predictions'] = predictions[0]
                    st.session_state['predicted_class'] = predicted_class
                    st.session_state['confidence'] = confidence
    
    with col2:
        st.header("Classification Results")
        
        # Display results
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            predicted_class = st.session_state['predicted_class']
            confidence = st.session_state['confidence']
            
            st.subheader("Prediction")
            
            if predicted_class == 'COVID-19':
                st.error(f"🦠 **{predicted_class}** (Confidence: {confidence:.2%})")
            elif predicted_class == 'Non-COVID':
                st.warning(f"⚠️ **{predicted_class}** (Confidence: {confidence:.2%})")
            else:
                st.success(f"✅ **{predicted_class}** (Confidence: {confidence:.2%})")
            
            # Display confidence scores for all classes
            st.subheader("Confidence Scores")
            
            for i, class_name in enumerate(class_names):
                confidence_score = float(predictions[i])
                st.write(f"**{class_name}**: {confidence_score:.2%}")
                st.progress(confidence_score)
            
            st.subheader("Probability Distribution")
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(class_names, predictions, color=['red', 'orange', 'green'])
            
            ax.set_ylabel('Probability')
            ax.set_title('Classification Probabilities')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2%}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Upload an image and click 'Classify Image' to see results.")
    
    # Add model information
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        **Model Details:**
        - **Architecture:** Convolutional Neural Network (CNN)
        - **Input Size:** 224x224 RGB images
        - **Classes:** COVID-19, Non-COVID, Normal
        - **Test Accuracy:** ~87.2%
        """)

if __name__ == "__main__":
    main() 