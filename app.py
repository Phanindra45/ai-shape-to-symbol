import streamlit as st
import cv2
import numpy as np
from PIL import Image
import uuid

st.set_page_config(page_title="Shape → Editable Symbol", layout="wide")
st.title("🧠 Shape → Editable Symbol (Correct Fill + Stroke)")

# -------------------------------------------------
# Extract shape mask & stroke edges
# -------------------------------------------------
def extract_shape_and_edges(image):
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ---- FILL MASK ----
    _, fill_mask = cv2.threshold(
        gray, 240, 255, cv2.THRESH_BINARY_INV
    )

    kernel = np.ones((7, 7), np.uint8)
    fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_CLOSE, kernel)

    # ---- STROKE EDGES ----
    edges = cv2.Canny(gray, 50, 150)

    return fill_mask, edges

# -------------------------------------------------
# Render symbol
# -------------------------------------------------
def render_symbol(
    fill_mask,
    edges,
    scale_x,
    scale_y,
    stroke_width,
    stroke_color,
    fill_color
):
    h, w = 600, 800
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    # HEX → BGR
    stroke_bgr = tuple(int(stroke_color[i:i+2], 16) for i in (5, 3, 1))
    fill_bgr   = tuple(int(fill_color[i:i+2], 16) for i in (5, 3, 1))

    # ---- FIND CONTOUR FOR FILL ----
    contours, _ = cv2.findContours(
        fill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnt = max(contours, key=cv2.contourArea).astype(np.float32)

    min_x, min_y = cnt.min(axis=(0, 1))
    cnt[:, 0, 0] = (cnt[:, 0, 0] - min_x) * scale_x + 150
    cnt[:, 0, 1] = (cnt[:, 0, 1] - min_y) * scale_y + 100
    cnt = cnt.astype(np.int32)

    # ---- FILL ----
    cv2.drawContours(canvas, [cnt], -1, fill_bgr, -1)

    # ---- STROKE (EDGE BASED) ----
    edge_points = np.column_stack(np.where(edges > 0))

    for y, x in edge_points:
        sx = int((x - min_x) * scale_x + 150)
        sy = int((y - min_y) * scale_y + 100)
        if 0 <= sx < w and 0 <= sy < h:
            cv2.circle(canvas, (sx, sy), stroke_width, stroke_bgr, -1)

    return canvas

# -------------------------------------------------
# UI
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload image", ["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, use_column_width=True)

    fill_mask, edges = extract_shape_and_edges(image)

    with col2:
        st.image(fill_mask, caption="Fill Mask", clamp=True)

    st.subheader("Edit Symbol")

    scale_x = st.slider("Scale X", 0.5, 3.0, 1.0)
    scale_y = st.slider("Scale Y", 0.5, 3.0, 1.0)
    stroke_width = st.slider("Stroke Width", 1, 6, 2)

    stroke_color = st.color_picker("Stroke Color", "#b44cff")
    fill_color = st.color_picker("Fill Color", "#ff0000")

    preview = render_symbol(
        fill_mask,
        edges,
        scale_x,
        scale_y,
        stroke_width,
        stroke_color,
        fill_color
    )

    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
             caption="Editable Symbol",
             use_column_width=True)