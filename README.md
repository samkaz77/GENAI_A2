# MAE Streamlit App

This project provides a Streamlit interface for the MAE (Masked Autoencoder) model defined in `2ndcopy.ipynb`.
It loads the trained checkpoint from:

- `results/outputs/mae_best.pt`

## Files

- `app.py`: Streamlit app for image reconstruction
- `requirements.txt`: Python dependencies
- `2ndcopy.ipynb`: Training notebook used to build the model
- `results/outputs/mae_best.pt`: Trained model checkpoint

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Usage

1. Open the Streamlit URL shown in terminal (usually `http://localhost:8501`).
2. Keep checkpoint path as default or set a custom path in the sidebar.
3. Upload an image (`jpg`, `jpeg`, `png`, `bmp`, or `webp`).
4. View:
   - Original image
   - Masked input
   - Reconstruction
5. Optionally adjust:
   - `Mask ratio`
   - `Random seed`

## Notes

- The app uses the same MAE architecture and preprocessing from the notebook.
- Preprocessing: resize to checkpoint image size + tensor conversion (no normalization).
- If `torch` is not installed, the app will not start.
