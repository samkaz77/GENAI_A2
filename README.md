# MAE Gradio Inference

This project provides a Gradio interface for performing inference using a pre-trained Masked Autoencoder Vision Transformer (MAE) model. The application allows users to upload images and see the reconstructed outputs based on the MAE model.

## Project Structure

```
mae-gradio-inference
├── src
│   ├── app_gradio.py        # Main application logic for loading the MAE model and Gradio interface
│   └── model.py     # Defines the MAEViT class and model functionalities
├── checkpoints
│   └── mae_best.pt          # Pre-trained model weights and configuration
├── requirements.txt          # Python dependencies for the 
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python src/app_gradio.py
   ```

4. **Access the Gradio interface:**
   Open your web browser and go to `http://localhost:7860` to interact with the application.

## Usage

- Upload an image using the provided interface.
- Adjust the mask ratio using the slider.
- Click on the "Reconstruct" button to see the masked input, reconstruction, and resized ground truth images.
