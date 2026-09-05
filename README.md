# Mask R-CNN Instance Segmentation

A Computer Vision project demonstrating **object detection and instance segmentation using a pretrained Mask R-CNN model with PyTorch and Torchvision**.

The project uses a pretrained **Mask R-CNN ResNet-50 FPN** model to detect objects in an image and generate an individual segmentation mask for each detected object.

---

## 📌 Project Overview

Unlike traditional image classification, which assigns a label to an entire image, instance segmentation identifies individual objects and determines the pixels belonging to each object.

This project demonstrates the complete inference pipeline:

```text
Input Image
     ↓
Image Preprocessing
     ↓
Pretrained Mask R-CNN
     ↓
Object Detection
     ↓
Bounding Boxes
     ↓
Class Labels
     ↓
Confidence Scores
     ↓
Instance Masks
     ↓
Visualization
```

The model produces:

- Object bounding boxes
- Object class labels
- Confidence scores
- Pixel-level instance masks

---

## 🎯 Objective

The main objective of this project is to explore **instance segmentation using Mask R-CNN** and understand how a pretrained deep learning model can simultaneously perform object detection and segmentation.

The project demonstrates how to:

- Load a pretrained Mask R-CNN model
- Process an input image
- Perform inference
- Extract bounding boxes
- Extract class labels
- Extract confidence scores
- Generate instance masks
- Apply confidence thresholding
- Visualize segmentation results

---

## 🧠 Model

The project uses:

**Mask R-CNN ResNet-50 FPN**

from the PyTorch Torchvision detection framework.

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn

model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
```

The pretrained model uses **COCO-trained weights** and can recognize multiple object categories.

The model architecture combines:

- ResNet-50 backbone
- Feature Pyramid Network (FPN)
- Region Proposal Network (RPN)
- ROI heads
- Bounding box prediction
- Classification
- Mask prediction

---

## 🏗️ Mask R-CNN Architecture

Mask R-CNN extends Faster R-CNN by adding an additional mask prediction branch.

```text
                 Input Image
                      │
                      ▼
               ResNet-50 Backbone
                      │
                      ▼
                Feature Pyramid
                   Network
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     Region Proposal          Feature Maps
       Network                    │
          │                       │
          └───────────┬───────────┘
                      ▼
                  ROI Heads
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
     Classification  Bounding    Mask
                      Boxes      Prediction
          │           │           │
          └───────────┼───────────┘
                      ▼
             Instance Segmentation
```

---

## 🔍 Instance Segmentation

Instance segmentation performs two tasks simultaneously:

### 1. Object Detection

The model identifies:

- What the object is
- Where the object is located

using:

```text
Class Label
Bounding Box
Confidence Score
```

### 2. Instance Segmentation

The model additionally predicts a pixel-level mask for every detected object.

This allows individual objects of the same class to be segmented separately.

---

## 📷 Input Image

The notebook uses:

```text
person_walking.jpeg
```

The image is loaded using Pillow:

```python
image = Image.open(image_path).convert("RGB")
```

---

## ⚙️ Image Preprocessing

The image is converted into a PyTorch tensor using:

```python
transform = T.Compose([
    T.ToTensor()
])

image_tensor = transform(image).unsqueeze(0)
```

The resulting tensor is then passed to the Mask R-CNN model.

---

## 🚀 Model Inference

Inference is performed without calculating gradients:

```python
with torch.no_grad():
    output = model(image_tensor)
```

This produces prediction results containing:

```text
boxes
labels
scores
masks
```

---

## 📊 Prediction Outputs

The model returns four important outputs:

| Output | Description |
|---|---|
| `boxes` | Bounding box coordinates |
| `labels` | Predicted object classes |
| `scores` | Confidence scores |
| `masks` | Pixel-level instance segmentation masks |

The project extracts these outputs using:

```python
boxes = output[0]["boxes"].cpu().numpy()
labels = output[0]["labels"].cpu().numpy()
masks = output[0]["masks"].cpu().numpy()
scores = output[0]["scores"].cpu().numpy()
```

---

## 🎚️ Confidence Threshold

A confidence threshold of:

```text
0.60
```

is used to filter predictions.

```python
threshold = 0.6

selected_indices = scores > threshold

boxes = boxes[selected_indices]
masks = masks[selected_indices]
```

Only predictions with confidence scores greater than 0.60 are retained for visualization.

---

## 🎭 Mask Processing

The predicted masks are converted into binary masks using a threshold of 0.5:

```python
masks = (masks > 0.5).squeeze(axis=1)
```

This converts the model's probability masks into binary segmentation masks.

---

## 🖼️ Visualization

For each detected object, the project:

1. Draws a bounding box.
2. Displays the predicted class label.
3. Creates a colored segmentation mask.
4. Overlays the mask onto the original image.

Example visualization pipeline:

```text
Original Image
      ↓
Detected Object
      ↓
Bounding Box
      +
Class Label
      +
Segmentation Mask
      ↓
Final Visualization
```

The output image is saved as:

```text
output_img.jpeg
```

---

## 🏷️ COCO Classes

The pretrained Mask R-CNN model uses the COCO object categories.

Some of the supported classes include:

- Person
- Bicycle
- Car
- Motorcycle
- Airplane
- Bus
- Train
- Truck
- Boat
- Bird
- Cat
- Dog
- Horse
- Sheep
- Cow
- Elephant
- Bear
- Zebra
- Giraffe
- Backpack
- Umbrella
- Bottle
- Chair
- Couch
- Laptop
- Cell Phone
- Book
- Scissors
- And many more

The notebook includes the COCO class-name mapping used to convert numerical labels into human-readable object names.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- Mask R-CNN
- ResNet-50
- Feature Pyramid Network (FPN)
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Google Colab
- CUDA / GPU

---

## 💻 Environment

The notebook is designed to run in a GPU-enabled environment such as Google Colab.

The notebook metadata specifies:

```text
Python
GPU Acceleration
NVIDIA Tesla T4
```

GPU acceleration can significantly improve inference performance for deep learning models.

---

## 📦 Installation

Install the required libraries:

```bash
pip install torch torchvision numpy opencv-python matplotlib pillow
```

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Farhana-Tani/Image-segmentation-using-Mask-RCNN.git
```

### 2. Open the Notebook

Open:

```text
Mask_RCNN.ipynb
```

You can run the notebook using:

- Google Colab
- Jupyter Notebook
- VS Code with Jupyter support

### 3. Prepare an Image

Place an image in the working directory or upload one to Google Colab.

Update the image path:

```python
image_path = "/content/person_walking.jpeg"
```

### 4. Load the Model

```python
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
```

### 5. Run Inference

```python
with torch.no_grad():
    output = model(image_tensor)
```

### 6. Process the Results

Extract:

```python
boxes
labels
scores
masks
```

### 7. Visualize the Results

Run the visualization section of the notebook.

The final result is displayed and saved as:

```text
output_img.jpeg
```

---

## 📂 Project Structure

```text
Mask-RCNN-Instance-Segmentation/
│
├── Mask_RCNN.ipynb
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🔬 Key Concepts Demonstrated

This project demonstrates several important Computer Vision concepts:

- Object detection
- Instance segmentation
- Bounding box prediction
- Pixel-level segmentation
- Transfer learning
- Pretrained deep learning models
- Feature Pyramid Networks
- Region Proposal Networks
- ROI-based prediction
- Confidence thresholding
- Image preprocessing
- Computer Vision visualization

---

## 🆚 Object Detection vs Instance Segmentation

| Task | Output |
|---|---|
| Image Classification | One or more class labels |
| Object Detection | Class + Bounding Box |
| Semantic Segmentation | Pixel-level class labels |
| Instance Segmentation | Pixel-level mask for each object |

Mask R-CNN performs **object detection and instance segmentation simultaneously**.

---

## 🌟 Why Mask R-CNN?

Mask R-CNN is particularly useful when bounding boxes alone are not sufficient.

For example:

```text
Object Detection:
"Where is the person?"

Instance Segmentation:
"Which exact pixels belong to this person?"
```

This makes instance segmentation useful for applications where precise object boundaries are important.

---

## 🚀 Applications

Mask R-CNN can be used in many Computer Vision applications, including:

- Medical image segmentation
- Cell and tissue segmentation
- Autonomous driving
- Robotics
- Human pose and activity analysis
- Industrial inspection
- Agriculture
- Wildlife monitoring
- Object counting
- Smart surveillance
- Scene understanding

---

## 🔬 Future Improvements

Possible improvements to this project include:

- Train Mask R-CNN on a custom dataset
- Fine-tune the model for a specific application
- Add quantitative evaluation using mAP
- Calculate mask IoU
- Add real-time webcam segmentation
- Add video instance segmentation
- Implement object tracking
- Compare Mask R-CNN with YOLO segmentation models
- Compare different backbone architectures
- Build a Streamlit interface
- Apply the model to medical imaging datasets

---

## 📚 Learning Outcomes

Through this project, I learned how to:

- Use pretrained Mask R-CNN models
- Perform instance segmentation with PyTorch
- Process model outputs
- Interpret bounding boxes and class labels
- Work with segmentation masks
- Apply confidence thresholds
- Visualize segmentation results
- Integrate PyTorch with OpenCV and Matplotlib
- Run deep learning inference using GPU acceleration

---

## 👩‍💻 Author

**Farhana Tani**

Computer Vision | Deep Learning | Artificial Intelligence

GitHub:

https://github.com/Farhana-Tani

---

## 📖 References

- PyTorch Torchvision Mask R-CNN documentation
- Mask R-CNN: He et al., 2017
- COCO Dataset
- Torchvision Object Detection and Instance Segmentation utilities

---

## 📜 License

This project is licensed under the MIT License.

See the `LICENSE` file for more information.
