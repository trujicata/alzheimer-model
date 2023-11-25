import streamlit as st
import torch
from models.classifier3D.model import Classifier3D
import h5py

from captum.attr import IntegratedGradients
from torchvision import transforms as T
import matplotlib.pyplot as plt


def main():
    st.title("Alzheimer's Disease Classification")

    with st.spinner("Loading model..."):
        model = Classifier3D()
        weights = torch.load(
            "data/best-model-so-far.ckpt", map_location=torch.device("cpu")
        )

        for k, v in weights.items():
            if k in model.state_dict().keys():
                model.state_dict()[k].copy_(v)
            else:
                print(f"Key {k} not found in model state dict")

        model.eval()

    with st.spinner("Loading data..."):
        h5_files = h5py.File("data/test_the_model/test.hdf5", "r")
        train_data = h5_files["X_nii"]
        labels = h5_files["y"]
        selected_file = 0

        to_tensor = T.ToTensor()

        input = train_data[selected_file]

        input = to_tensor(input).unsqueeze(0).unsqueeze(0)
        label = model.classes[int(labels[selected_file])]
        baseline = torch.zeros_like(input)

    if "predictions" not in st.session_state.keys():
        with st.spinner("Calculating predictions..."):
            preds = model.predict(input)
            st.session_state["predictions"] = preds

    st.table(st.session_state["predictions"])
    st.write(f"Label: {label}")

    if "attributions" not in st.session_state.keys():
        with st.spinner("Calculating attributions..."):
            ig = IntegratedGradients(model)
            attributions, delta = ig.attribute(
                input, baseline, target=0, return_convergence_delta=True
            )
            st.session_state["attributions"] = attributions.detach().numpy()
    index = st.slider("Slice", 0, 120, 1)

    toggler = st.checkbox("Show attributions")
    fig, ax = plt.subplots()
    if toggler:
        ax.imshow(
            st.session_state["attributions"][0, 0, index, :, :],
            cmap="bwr",
        )
    ax.imshow(input[0, 0, index, :, :], alpha=0.5, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)


if __name__ == "__main__":
    main()
