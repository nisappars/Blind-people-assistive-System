import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from collections import defaultdict
import pickle
import torch
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch.nn as nn



model_path = "color_classifier.pth"
label_encoder_path = "label_encoder.pkl"
csv_file = "colors.csv"

class ColorClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ColorClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_image_colors(image, model, label_encoder):
    pixels = np.array(image.getdata()) / 255.0
    pixels = torch.tensor(pixels, dtype=torch.float32)

    with torch.no_grad():
        model.eval()
        outputs = model(pixels)
        _, predicted_indices = torch.max(outputs, 1)
        color_names = label_encoder.inverse_transform(predicted_indices.numpy())

    color_counter = defaultdict(int)
    for color_name in color_names:
        color_counter[color_name] += 1

    return color_counter

def get_rgb_from_csv(colors_list, csv_file_path):
    df = pd.read_csv(csv_file_path)
    colors_dict = {row["Name"].lower(): (int(row["Red (8 bit)"]), int(row["Green (8 bit)"]), int(row["Blue (8 bit)"])) for _, row in df.iterrows()}
    result = []
    for color_name in colors_list:
        if color_name.lower() in colors_dict:
            rgb = colors_dict[color_name.lower()]
            result.append((color_name, rgb))
        else:
            result.append((color_name, None))
    return result

def rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"

def adjust_brightness_if_black_is_high(image, threshold=0.02, factor=1.5):
    pixels = np.array(image.getdata())
    black_pixel_count = np.sum(np.all(pixels == 0, axis=1))
    total_pixel_count = len(pixels)
    black_ratio = black_pixel_count / total_pixel_count

    if black_ratio > threshold:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)

    return image, black_ratio > threshold

st.title("Audio Description")

# Capture image from camera
image_file = st.camera_input("Take a picture")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if image_file or uploaded_file is not None:
      
    if(image_file is not None): image = Image.open(image_file)
    else: image = Image.open(uploaded_file)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    st.write("Original Image:")
    st.image(image, caption='Original Image', use_column_width=True)
    
    image, adjusted = adjust_brightness_if_black_is_high(image)

    if adjusted:
        st.write("Brightened Image (after adjustment):")
        st.image(image, caption='Brightened Image', use_column_width=True)

    st.write('Processing...')
    processor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
    model = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process(outputs, target_sizes=target_sizes)[0]
    # Sort the results based on score and box size
    #sorted_results = sorted(zip(results["scores"], results["labels"], results["boxes"]), key=lambda x: (x[0], -((x[2][2] - x[2][0]) * (x[2][3] - x[2][1]))), reverse=True)

    # Filter the results with score above 0.5 and delete others
    #filtered_results = [(score, label, box) for score, label, box in sorted_results if score > 0.5]

    # Update the results with the filtered results
    # if filtered_results:
    #     results["scores"], results["labels"], results["boxes"] = zip(*filtered_results)

    draw = ImageDraw.Draw(image)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # draw.rectangle(box, outline="red", width=3)
        # draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", fill="red")
        break
    # st.write("Detected Objects:")
    # st.image(image, caption='Detected Objects', use_column_width=True)

    # Load the color classifier model and label encoder
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
        num_classes = len(label_encoder.classes_)

    color_model = ColorClassifier(3, num_classes)
    color_model.load_state_dict(torch.load(model_path))

    # Perform color recognition on detected objects
    for box in results["boxes"]:
        box = [round(i) for i in box.tolist()]
        cropped_image = image.crop(box)

        # Display cropped image
        #st.image(cropped_image, caption='Cropped Object', use_column_width=True)

        # Get image colors
        image_colors = get_image_colors(cropped_image, color_model, label_encoder)

        # Get the top 5 colors
        color_pixels = sorted(image_colors.items(), key=lambda x: x[1], reverse=True)
        color_pixels = color_pixels[:5]
        color_codes = get_rgb_from_csv([color[0] for color in color_pixels], csv_file)

        # Display results
        st.write("Top 5 Colors:")
        color_names = []
        hex_values = []
        pixel_counts = []
        for (color_name, rgb_val), (_, pixels) in zip(color_codes, color_pixels):
            hex_val = rgb_to_hex(rgb_val) if rgb_val else "N/A"
            st.write({
                "Color name": color_name,
                "RGB value": rgb_val if rgb_val else "N/A",
                "Hex value": hex_val,
                "Pixel count": pixels
            })
            color_names.append(color_name)
            hex_values.append(hex_val)
            pixel_counts.append(pixels)

        # Reverse the order for plotting
        color_names.reverse()
        hex_values.reverse()
        pixel_counts.reverse()

        # Plot the graph
        fig, ax = plt.subplots()
        bars = ax.barh(color_names, pixel_counts, color=hex_values)
        ax.set_xlabel('Pixel Count')
        ax.set_title('Top 5 Colors in Object')

        # Add color hex labels to the bars
        for bar, hex_val in zip(bars, hex_values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f' {hex_val}', va='center')

        st.pyplot(fig)
        break
