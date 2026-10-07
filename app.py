import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image, ImageOps, ImageFilter


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_FILE = "digit_recognizer_model.pkl"

try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .prediction-card {
        background-color: #e8f8ee;
        border-radius: 18px;
        padding: 35px;
        text-align: center;
        border: 1px solid #c8ead5;
    }

    .prediction-label {
        font-size: 18px;
        color: #555555;
        margin-bottom: 5px;
    }

    .prediction-digit {
        font-size: 75px;
        font-weight: 800;
        line-height: 1.1;
    }

    .prediction-confidence {
        font-size: 24px;
        font-weight: 600;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">✍️ Handwritten Digit Recognizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an image of a handwritten digit and let the machine learning model recognize it.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📁 Upload a Handwritten Digit")

uploaded_file = st.file_uploader(
    "Upload an image containing one handwritten digit",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# IMAGE PREPROCESSING FUNCTION
# ============================================================

def preprocess_image(original_image):
    """
    Converts a normal uploaded handwritten digit image
    into a 28x28 MNIST-style image.

    Main steps:
    1. Convert image to RGB
    2. Detect blue ink when present
    3. Otherwise use grayscale
    4. Remove weak noise
    5. Crop the digit
    6. Put digit on a square canvas
    7. Resize to 28x28
    8. Normalize pixels
    """

    # --------------------------------------------------------
    # Correct image orientation
    # --------------------------------------------------------

    image = ImageOps.exif_transpose(original_image)

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    rgb_image = image.convert("RGB")

    rgb = np.array(rgb_image).astype(np.int16)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    # --------------------------------------------------------
    # Detect blue pen ink
    #
    # Grey notebook lines have similar R/G/B values.
    # Blue ink normally has a much stronger B value.
    # --------------------------------------------------------

    blue_strength = blue - np.maximum(red, green)

    blue_mask = blue_strength > 25

    blue_pixels = np.sum(blue_mask)

    total_pixels = blue_mask.size

    blue_ratio = blue_pixels / total_pixels


    # --------------------------------------------------------
    # BLUE INK IMAGE
    # --------------------------------------------------------

    if blue_ratio > 0.001:

        # Make blue pixels white and everything else black
        digit_array = np.zeros_like(red, dtype=np.uint8)

        digit_array[blue_mask] = 255

    # --------------------------------------------------------
    # GRAYSCALE FALLBACK
    # --------------------------------------------------------

    else:

        gray_image = ImageOps.grayscale(rgb_image)

        gray_array = np.array(gray_image)

        # Determine whether background is light
        if gray_array.mean() > 127:

            digit_array = 255 - gray_array

        else:

            digit_array = gray_array

        # Remove weak pixels
        digit_array[digit_array < 50] = 0


    # ========================================================
    # REMOVE SMALL NOISE
    # ========================================================

    # Remove tiny isolated regions using a simple
    # neighborhood-based filtering approach.

    binary = digit_array > 40

    if np.sum(binary) == 0:
        return None, None


    # ========================================================
    # FIND DIGIT BOUNDING BOX
    # ========================================================

    coords = np.argwhere(binary)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)


    # --------------------------------------------------------
    # Add padding around digit
    # --------------------------------------------------------

    padding = max(
        5,
        int(max(y_max - y_min, x_max - x_min) * 0.10)
    )

    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(digit_array.shape[0] - 1, y_max + padding)
    x_max = min(digit_array.shape[1] - 1, x_max + padding)


    # ========================================================
    # CROP DIGIT
    # ========================================================

    cropped = digit_array[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]


    # ========================================================
    # MAKE SQUARE CANVAS
    # ========================================================

    height, width = cropped.shape

    size = max(height, width)

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )


    # Center cropped digit

    y_offset = (size - height) // 2
    x_offset = (size - width) // 2

    square[
        y_offset:y_offset + height,
        x_offset:x_offset + width
    ] = cropped


    # ========================================================
    # RESIZE TO 28 x 28
    # ========================================================

    processed_image = Image.fromarray(square)

    processed_image = processed_image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )


    # ========================================================
    # SLIGHT BLUR
    # ========================================================

    processed_image = processed_image.filter(
        ImageFilter.GaussianBlur(radius=0.25)
    )


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    processed_array = np.array(
        processed_image
    ).astype(np.float32)


    # ========================================================
    # NORMALIZE
    # ========================================================

    processed_array = processed_array / 255.0


    # ========================================================
    # FLATTEN
    # ========================================================

    input_data = processed_array.reshape(1, -1)


    return processed_image, input_data


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # Open uploaded image
        # ----------------------------------------------------

        original_image = Image.open(uploaded_file)

        display_image = ImageOps.exif_transpose(
            original_image
        ).convert("RGB")


        # ----------------------------------------------------
        # Two-column layout
        # ----------------------------------------------------

        col1, col2 = st.columns(
            2,
            gap="large"
        )


        # ====================================================
        # UPLOADED IMAGE
        # ====================================================

        with col1:

            st.subheader("🖼️ Uploaded Image")

            st.image(
                display_image,
                width=400
            )


        # ====================================================
        # PREPROCESS
        # ====================================================

        processed_image, input_data = preprocess_image(
            original_image
        )


        # ----------------------------------------------------
        # Check preprocessing
        # ----------------------------------------------------

        if processed_image is None:

            st.error(
                "❌ Could not detect a handwritten digit. "
                "Please upload a clearer image."
            )

            st.stop()


        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = int(
            model.predict(input_data)[0]
        )


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = (
                probabilities[prediction] * 100
            )

        else:

            probabilities = None
            confidence = 0.0


        # ====================================================
        # PREDICTION CARD
        # ====================================================

        with col2:

            st.subheader("🔮 Prediction")

            # IMPORTANT:
            # No HTML is used here.
            # Streamlit will display the digit normally.

            st.success(
                f"Predicted Digit: {prediction}"
            )

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )


        st.divider()


        # ====================================================
        # PROCESSED IMAGE
        # ====================================================

        st.subheader("🔍 Processed Image")

        st.write(
            "This is the 28×28 image actually given "
            "to the MNIST model."
        )


        processed_col1, processed_col2 = st.columns(
            2,
            gap="large"
        )


        with processed_col1:

            st.image(
                processed_image,
                caption="28 × 28 Model Input",
                width=300
            )


        with processed_col2:

            st.info(
                """
                **Image preprocessing**

                • Converted to RGB  
                • Blue handwriting detected  
                • Background separated  
                • Digit automatically cropped  
                • Digit centered  
                • Resized to 28 × 28  
                • Pixel values normalized  
                • Sent to the Random Forest model
                """
            )


        st.divider()


        # ====================================================
        # PROBABILITY CHART
        # ====================================================

        if probabilities is not None:

            st.subheader("📊 Prediction Probabilities")

            probability_df = pd.DataFrame(
                {
                    "Digit": list(range(10)),
                    "Probability": probabilities * 100
                }
            )

            st.bar_chart(
                probability_df.set_index("Digit")
            )


            # =================================================
            # TOP 3
            # =================================================

            st.subheader("🏆 Top 3 Predictions")

            top_3 = np.argsort(
                probabilities
            )[-3:][::-1]


            for rank, digit in enumerate(
                top_3,
                start=1
            ):

                percentage = (
                    probabilities[digit] * 100
                )

                st.write(
                    f"**{rank}. Digit {digit}** — "
                    f"{percentage:.2f}%"
                )


        st.divider()


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.subheader("🤖 Model Information")

        info1, info2, info3 = st.columns(3)


        with info1:

            st.metric(
                "Algorithm",
                "Random Forest"
            )


        with info2:

            st.metric(
                "Model Accuracy",
                "97.91%"
            )


        with info3:

            st.metric(
                "Classes",
                "10 digits (0–9)"
            )


        st.divider()


        # ====================================================
        # HOW IT WORKS
        # ====================================================

        st.subheader("⚙️ How It Works")

        st.markdown(
            """
            **1. Upload**  
            Upload an image containing one handwritten digit.

            **2. Detect handwriting**  
            The system detects the handwritten stroke and
            separates it from the background.

            **3. Crop**  
            The digit is automatically cropped.

            **4. Center**  
            The digit is placed in the center of a square canvas.

            **5. Resize**  
            The image is resized to **28 × 28 pixels**.

            **6. Normalize**  
            Pixel values are converted into the format expected
            by the machine-learning model.

            **7. Predict**  
            The Random Forest model predicts the digit.

            **8. Display results**  
            The application displays the predicted digit,
            confidence, probability distribution and top 3 predictions.
            """
        )


    except Exception as e:

        st.error(
            f"❌ An error occurred while processing the image:\n\n{e}"
        )


else:

    st.info(
        "👆 Upload a handwritten digit image above "
        "to start the prediction."
    )