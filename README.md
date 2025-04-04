# Assistive System for Blind People

This project is an assistive system that processes images, detects objects, and identifies their colors to provide an audio description, aiding visually impaired individuals.

## Features
- Capture or upload an image.
- Adjust brightness if the image is too dark.
- Detect objects using **YOLOS Fashionpedia Model**.
- Recognize and classify colors of detected objects.
- Display results with visual and text descriptions.
- Plot a bar chart of detected colors with their RGB and Hex values.

## Tech Stack
- **Python**
- **Streamlit** (for UI)
- **PyTorch** (for object detection and color classification)
- **Pandas** (for CSV data handling)
- **PIL (Pillow)** (for image processing)
- **Matplotlib** (for visualization)
- **Transformers** (for YOLOS model)

## Installation
### 1. Clone the Repository
```sh
git clone https://github.com/<your-username>/Assistive-System.git
cd Assistive-System
```
### 2. Install Dependencies
```sh
pip install -r requirements.txt
```
### 3. Run the Application
```sh
streamlit run streamlit.py
```

## Project Structure
```
Assistive-System/
│── color_classifier.pth      # Trained PyTorch model for color classification
│── label_encoder.pkl         # Encoded labels for color classification
│── colors.csv                # Dataset for mapping colors
│── streamlit.py              # Main application script
│── README.md                 # Documentation
│── requirements.txt          # List of dependencies
```

## Usage
1. Open the application using **Streamlit**.
2. Capture an image using your webcam or upload an existing image.
3. The system will process the image, detect objects, and identify their colors.
4. The results will be displayed with color names, RGB, and Hex values along with a bar chart.

## Future Enhancements
- Add text-to-speech support for audio descriptions.
- Improve object detection accuracy.
- Expand the color dataset for better recognition.

## License
This project is open-source and available under the **MIT License**.

## Contributors
- Nishanth Arun T.
