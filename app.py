import streamlit as st
import cv2
import numpy as np
from PIL import Image
import uuid

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="AI Shape → Editable Symbol Converter",
    layout="wide"
)

st.title("🧠 AI Shape → Editable Symbol Converter (Stroke + Fill)")

# -------------------------------------------------
# Extract Freeform Contour
# -------------------------------------------------
def extract_contour(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, edges

    largest = max(contours, key=cv2.contourArea)
    return largest, edges

# -------------------------------------------------
# Render Symbol with FILL + STROKE
# -------------------------------------------------
def render_symbol(
    contour,
    scale_x,
    scale_y,
    stroke_width,
    stroke_color,
    fill_color
):
    canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255
    contour = contour.astype(np.float32)

    # Normalize contour to origin
    contour -= contour.min(axis=0)

    # Apply scaling
    contour[:, 0, 0] *= scale_x
    contour[:, 0, 1] *= scale_y

    # Center on canvas
    contour[:, 0, 0] += 100
    contour[:, 0, 1] += 100

    contour_i = contour.astype(np.int32)

    # Convert colors HEX → BGR
    stroke_rgb = tuple(int(stroke_color[i:i+2], 16) for i in (1, 3, 5))
    fill_rgb = tuple(int(fill_color[i:i+2], 16) for i in (1, 3, 5))

    stroke_bgr = stroke_rgb[::-1]
    fill_bgr = fill_rgb[::-1]

    # ---- FILL INSIDE SHAPE ----
    cv2.drawContours(
        canvas,
        [contour_i],
        -1,
        fill_bgr,
        thickness=-1
    )

    # ---- DRAW STROKE OUTLINE ----
    cv2.drawContours(
        canvas,
        [contour_i],
        -1,
        stroke_bgr,
        thickness=int(stroke_width)
    )

    return canvas

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Shape Image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    contour, edges = extract_contour(image)

    with col2:
        st.subheader("Edge Detection")
        st.image(edges, use_column_width=True)

    if contour is not None:
        st.success("Detected Shape Type: Freeform")

        # -------------------------
        # Editing Controls
        # -------------------------
        st.subheader("✏️ Edit Symbol")

        scale_x = st.slider(
            "Width (Scale X)",
            0.5, 3.0, 1.0
        )

        scale_y = st.slider(
            "Height (Scale Y)",
            0.5, 3.0, 1.0
        )

        stroke_width = st.slider(
            "Stroke Width",
            1, 10, 2
        )

        stroke_color = st.color_picker(
            "Stroke Color (Outline)",
            "#000000"
        )

        fill_color = st.color_picker(
            "Fill Color (Inside Symbol)",
            "#FFFFFF"
        )

        # -------------------------
        # Render Preview
        # -------------------------
        preview = render_symbol(
            contour,
            scale_x,
            scale_y,
            stroke_width,
            stroke_color,
            fill_color
        )

        st.subheader("🧩 Editable Symbol Preview")
        st.image(preview, use_column_width=True)

        # -------------------------
        # Symbol Data Model
        # -------------------------
        symbol = {
            "symbol_id": "sym_" + uuid.uuid4().hex[:6],
            "symbol_type": "Freeform",
            "parameters": {
                "scale_x": float(scale_x),
                "scale_y": float(scale_y)
            },
            "properties": {
                "strokeWidth": int(stroke_width),
                "strokeColor": stroke_color,
                "fillColor": fill_color
            }
        }

        st.subheader("📦 Symbol Data Model")
        st.json(symbol)

    else:
        st.error("No shape detected. Please upload a clear shape image.")