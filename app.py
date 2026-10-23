
import cv2
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms

st.set_page_config(
    page_title="Chest X-ray AI",
    page_icon="🩻",
    layout="wide",
)

MODEL_PATH = Path(
    "/Volumes/Disk externe/NIH-ChestXray14/"
    "outputs-3-epochs/best_model.pt"
)

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding",
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(
        model.classifier.in_features, len(LABELS)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def gradcam(model, tensor, class_index):
    """Crée une carte Grad-CAM pour une classe prédite."""
    activations = []
    gradients = []
    target_layer = model.features.denseblock4

    def save_activation(module, inputs, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle1 = target_layer.register_forward_hook(save_activation)
    handle2 = target_layer.register_full_backward_hook(save_gradient)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        logits[0, class_index].backward()

        feature_maps = activations[0][0]
        grads = gradients[0][0]

        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * feature_maps).sum(dim=0)
        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam
    finally:
        handle1.remove()
        handle2.remove()


def make_overlay(image, cam):
    # Redimensionne la carte thermique à la taille de la radio.
    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    ).resize(image.size, Image.Resampling.BILINEAR)

    heat = np.asarray(cam_image).astype(np.float32) / 255.0

    # Palette de chaleur : bleu/rouge/jaune selon l'intensité.
    heat_rgb = plt.get_cmap("jet")(heat)[..., :3]
    heat_rgb = np.uint8(heat_rgb * 255)

    overlay = Image.blend(
        image.convert("RGB"),
        Image.fromarray(heat_rgb),
        alpha=0.38,
    )
    return overlay


def draw_gradcam_contours(original_image, cam, threshold_percentile=85):
    # Convertir la radio en tableau RGB
    original = np.array(original_image.convert("RGB"))

    # Agrandir Grad-CAM à la taille de la radio
    cam_resized = cv2.resize(
        cam.astype(np.float32),
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # Normaliser les valeurs entre 0 et 1
    cam_resized -= cam_resized.min()
    if cam_resized.max() > 0:
        cam_resized /= cam_resized.max()

    # Sélectionner les zones les plus activées
    threshold = np.percentile(cam_resized, threshold_percentile)
    mask = (cam_resized >= threshold).astype(np.uint8) * 255

    # Nettoyer le masque
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Détecter et dessiner les contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    result = original.copy()
    cv2.drawContours(result, contours, -1, (255, 0, 0), 3)

    return Image.fromarray(result), Image.fromarray(mask)


st.title("🩻 Chest X-ray AI")
st.write(
    "Classification expérimentale et visualisation des régions "
    "qui influencent la prédiction."
)

st.warning(
    "Recherche uniquement — cette visualisation ne constitue pas "
    "un diagnostic médical. Grad-CAM indique des zones d'influence, "
    "pas les contours exacts d'une lésion."
)

uploaded = st.file_uploader(
    "Importer une radiographie thoracique",
    type=["png", "jpg", "jpeg"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Radiographie originale",
             use_container_width=True)

    if st.button("Analyser et localiser", type="primary"):
        try:
            model = load_model()
            tensor = transform(image).unsqueeze(0)

            # Grad-CAM a besoin des gradients, même en mode évaluation.
            cam = None
            with torch.enable_grad():
                logits = model(tensor)
                probabilities = torch.sigmoid(logits)[0]
                scores = probabilities.detach().cpu().numpy()
                best_index = int(np.argmax(scores))

                cam = gradcam(model, tensor, best_index)

            overlay = make_overlay(image, cam)

            contour_image, contour_mask = draw_gradcam_contours(image, cam)

            
            st.subheader("Scores prédits par le modèle")

            st.caption(
                "Scores expérimentaux : ils ne représentent pas des "
                "probabilités médicales validées."
            )

            results = sorted(
                zip(LABELS, scores),
                key=lambda item: item[1],
                reverse=True,
            )

            for label, score in results:
                st.write(f"**{label.replace('_', ' ')}** : {score:.1%}")


                

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    image,
                    caption="Radiographie originale",
                    use_container_width=True,
                )

            with col2:
                st.image(
                    overlay,
                    caption="Carte thermique Grad-CAM",
                    use_container_width=True,
                )

            with col3:
                st.image(
                    contour_image,
                    caption="Contours indicatifs en rouge",
                    use_container_width=True,
                )

            with st.expander("Afficher le masque des zones activées"):
                st.image(
                    contour_mask,
                    caption="Masque dérivé de Grad-CAM",
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"Erreur pendant l'analyse : {error}")
