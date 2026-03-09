import json
import numpy as np
import streamlit as st
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetV2B0
from pathlib import Path
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
DROPOUT = 0.4
L2_REG = 1e-4
WEIGHTS_PATH = Path("exported_model/sparepart_classifier.weights.h5")
CLASS_NAMES_PATH = Path("exported_model/class_names.json")


@st.cache_resource
def load_model():
    """Rebuild architecture and load saved weights."""
    with open(CLASS_NAMES_PATH) as f:
        class_map = json.load(f)
    class_names = [class_map[str(i)] for i in range(len(class_map))]
    num_classes = len(class_names)

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    base = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        include_preprocessing=True,
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(DROPOUT, name="dropout_1")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="dense_256",
    )(x)
    x = layers.Dropout(DROPOUT / 2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="sparepart_classifier")
    model.load_weights(str(WEIGHTS_PATH))
    return model, class_names


def predict(model, class_names, image: Image.Image) -> dict:
    """Run inference on a PIL image and return class probabilities."""
    img = image.resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    if x.ndim == 2:  # grayscale
        x = np.stack([x] * 3, axis=-1)
    elif x.shape[-1] == 4:  # RGBA
        x = x[:, :, :3]
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    return {name: float(probs[i]) for i, name in enumerate(class_names)}


# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spare Part Recognition", layout="wide")
st.title("Spare Part Recognition")

model, class_names = load_model()

input_mode = st.radio("Input method", ["Camera", "Upload"], horizontal=True)

col_left, col_right = st.columns(2)

with col_left:
    st.header("Image")
    source_image = None

    if input_mode == "Camera":
        camera_image = st.camera_input("Take a photo of a spare part")
        if camera_image is not None:
            source_image = camera_image
    else:
        uploaded_file = st.file_uploader(
            "Upload a spare part image", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        if uploaded_file is not None:
            source_image = uploaded_file
            st.image(uploaded_file, use_container_width=True)

with col_right:
    st.header("Result")

    if source_image is not None:
        image = Image.open(source_image).convert("RGB")
        results = predict(model, class_names, image)

        predicted_class = max(results, key=results.get)
        confidence = results[predicted_class]

        st.success(f"**Predicted: {predicted_class}** — {confidence:.1%}")

        st.subheader("Class Probabilities")
        for name in sorted(results, key=results.get, reverse=True):
            pct = results[name]
            st.markdown(f"**{name}**")
            st.progress(pct, text=f"{pct:.1%}")
    else:
        st.info("Take a photo or upload an image to get a prediction.")
