import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Set Tesseract CMD if required
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.title("re-imagine your pictures using ai preprocessing technics")

# Image upload
uploaded_image = st.file_uploader("upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="uploaded image", use_container_width=True)

    # Convert to OpenCV format
    image = np.array(image)

    # Preprocessing methods
    def preprocess_image(method):
        if method == "Original":
            return image
        elif method == "Grayscale":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif method == "Sharpened":
            kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel_sharpening)
        elif method == "Edge Detected":
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.Canny(grayscale, 100, 200)
        elif method == "Blurred":
            return cv2.GaussianBlur(image, (11, 11), 0)
        elif method == "Deblurred":
            kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel_sharpening)
        elif method == "Thresholded":
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, threshold_image = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY)
            return threshold_image
        elif method == "Adaptive Thresholding":
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif method == "Histogram Equalized":
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.equalizeHist(grayscale)
        elif method == "Inverted":
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.bitwise_not(grayscale)
        elif method == "Sobel Edge Detection":
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = cv2.magnitude(sobelx, sobely)
            return np.uint8(sobel_combined)
        elif method == "Median Blurred":
            return cv2.medianBlur(image, 5)
        elif method == "CLAHE":
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(grayscale)

    # Dropdown menu for selecting preprocessing methods
    methods = [
        "Original",
        "Grayscale",
        "Sharpened",
        "Edge Detected",
        "Blurred",
        "Deblurred",
        "Thresholded",
        "Adaptive Thresholding",
        "Histogram Equalized",
        "Inverted",
        "Sobel Edge Detection",
        "Median Blurred",
        "CLAHE"
    ]

    selected_method = st.selectbox("select preprocessing method", methods)

    # Display the selected preprocessed image
    processed_image = preprocess_image(selected_method)
    st.image(processed_image, caption=f"{selected_method} Image", use_container_width=True, channels="GRAY" if selected_method != "Original" else "RGB")

    # Perform OCR if the image contains text
    st.subheader("extracted text")
    ocr_result = pytesseract.image_to_string(image)

    if ocr_result.strip():
        st.text_area("extracted text", value=ocr_result, height=200)

        # Add download button for extracted text
        st.download_button(
            label="download extracted text",
            data=ocr_result,
            file_name="extracted_text.txt",
            mime="text/plain"
        )
    else:
        st.info("no text detected in the uploaded image.")

st.markdown(
        """
        **app by:** [rajaguru irusan](https://www.linkedin.com/in/rajaguruirusan)
        
        - **follow me in** [GitHub](https://github.com/rajaguruirusan)
        """
    )
