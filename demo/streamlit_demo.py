import streamlit as st
import torch
from models.resnet.model import Classifier3D as Model
import h5py
from captum.attr import IntegratedGradients
from torchvision import transforms as T
import matplotlib.pyplot as plt
import pandas as pd
import masking
import os
import torch.nn.functional as F


def main():
    st.title("Alzheimer's Disease Classification")

    # Add a file uploader for selecting the HDF5 file
    uploaded_file = st.text_input("Enter the path to the HDF5 file:")
    if uploaded_file and os.path.isfile(uploaded_file):
        with st.spinner("Loading model..."):
            model = Model(name="ResNet")
            checkpoint_path = "demo/pesos_resnet.ckpt"
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
            state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
            model.load_state_dict(state_dict)
            model.model.eval()

        with st.spinner("Loading data..."):
            h5_files = h5py.File(uploaded_file, "r")
            train_data = h5_files["X_nii"]
            labels = h5_files["y"]

            # Add a slider to select the file number
            selected_file = st.slider("Select file number", 0, len(train_data) - 1, 30)

            to_tensor = T.ToTensor()

            input = train_data[selected_file]
            input = to_tensor(input).unsqueeze(0).unsqueeze(0)
            label = model.classes[int(labels[selected_file])]
            baseline = torch.zeros_like(input)

        if "predictions" not in st.session_state.keys() or st.session_state.get("selected_file") != selected_file:
            with st.spinner("Calculating predictions..."):
                with torch.no_grad():
                    output = model(input)
                    predicted = output.argmax(dim=1)
                    probabilities = F.softmax(output, dim=1)
                    preds = model.classes[predicted.item()]
                    st.session_state["probabilities"] = probabilities.squeeze().tolist()
                    st.session_state["predictions"] = preds
                    st.session_state["selected_file"] = selected_file

        # Display the probabilities for each class
        st.write("Class Probabilities:")
        probabilities = st.session_state["probabilities"]
        predicted_class = st.session_state["predictions"]
        for i, prob in enumerate(probabilities):
            class_name = model.classes[i]
            if class_name == predicted_class:
                st.markdown(f"<span style='color:red'>{class_name}: {prob:.4f}</span>", unsafe_allow_html=True)
            else:
                st.write(f"{class_name}: {prob:.4f}")

        # Display the predicted label
        st.write(f"Predicted label: {st.session_state['predictions']}")
        st.write(f"True label: {label}")

        area_labels = pd.read_csv('demo/CerebrA_LabelDetails.csv')

        # Initialize attributions
        attributions = None

        if "attributions" not in st.session_state.keys() or st.session_state.get("selected_file") != selected_file:
            with st.spinner("Calculating attributions..."):
                ig = IntegratedGradients(model)
                attributions, delta = ig.attribute(
                    input, baseline, target=0, return_convergence_delta=True
                )
                st.session_state["attributions"] = attributions.detach().numpy()
                st.session_state["selected_file"] = selected_file
        else:
            attributions = torch.tensor(st.session_state["attributions"])

        index = st.slider("Slice", 0, 120, 1)

        toggler = st.checkbox("Show attributions", value=True)
        fig, ax = plt.subplots()
        if toggler:
            ax.imshow(
                st.session_state["attributions"][0, 0, index, :, :],
                cmap="bwr",
            )
        ax.imshow(input[0, 0, index, :, :], alpha=0.5, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

        with st.spinner("Calculating intensity values..."):
            intensity_values = masking.process_image(input, attributions, area_labels)
            st.session_state["intensity_values"] = intensity_values

        st.dataframe(st.session_state["intensity_values"])
    else:
        st.warning("Please enter a valid file path.")

if __name__ == "__main__":
    main()