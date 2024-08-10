import streamlit as st
import numpy as np
from PIL import Image
import pickle
import cv2
import sklearn
from sklearn.tree import DecisionTreeClassifier



# Display title
image_path = 'innomatics-logo-img.png'  # Replace with your actual PNG image file path

# Display the PNG image
st.image(image_path)

# Load the pre-trained model
#model_path = "potato_plant_disease.pkl"
#model = pickle.load(model_path)


file_path = r'C:\Users\Administrator\MachineLearning\potato_file\potato_plant_disease_dtc.pkl'
#new
with open(file_path , 'rb') as f:
    model = pickle.load(f)


# Function to preprocess the image
def preprocess_image(image):
    try:
        # Resize the image to (256, 256) as per your model's input requirement
        resized_image = image.resize((256, 256))
        # Convert to numpy array and flatten
        img_array = np.array(resized_image)
        # Check if the image has three color channels (RGB), if so, convert it to grayscale
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        flattened_img = img_array.flatten()
        # Ensure the flattened image has exactly 400 features
        if flattened_img.shape[0] != 65536:
            raise ValueError(f"Expected 65536 features, but got {flattened_img.shape[0]} features.")
        processed_image = flattened_img.reshape(1, -1)
        prediction = model.predict(processed_image)[0]
        return prediction
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Streamlit app
def main():
    st.title('Potato Plant Disease')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])  # Adjust type as per your image types

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            if st.button('Submit'):
                # Preprocess the image
                predicted_image = preprocess_image(image)
                st.write(f'Raw Prediction: {predicted_image}')
                    
        
        except Exception as e:
            st.error(f"Error processing or classifying image: {e}")

if __name__ == '__main__':
    main()