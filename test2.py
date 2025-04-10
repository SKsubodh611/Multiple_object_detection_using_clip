#response1 , good image quality , slow but ViT-B/16 
import cv2
import torch
import clip
import numpy as np
from PIL import Image

# Load CLIP model (faster ViT-B/16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# Define objects to detect
object_descriptions = ["a person", "a dog", "a cat", "a cap", "a car", "a tree", "a chair",
                       "a football", "a cup", "a laptop", "a phone"]

# Pre-tokenize and encode text descriptions once
text_inputs = torch.cat([clip.tokenize(desc) for desc in object_descriptions]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Selective Search Initialization
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

frame_count = 0
process_every_n_frames = 3  # Only process every Nth frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        continue

    # Resize frame to reduce processing load
    frame = cv2.resize(frame, (144, 144))

    # Apply Selective Search
    ss.setBaseImage(frame)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    proposals = []
    coordinates = []

    for (x, y, w, h) in rects[:20]:  # Limit proposals
        if w < 60 or h < 60 or w/h > 2.5 or h/w > 2.5:
            continue  # Skip bad shapes

        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (224, 224))  # CLIP input size

        pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        image_tensor = preprocess(pil_image).unsqueeze(0)  # shape: [1, 3, 224, 224]

        proposals.append(image_tensor)
        coordinates.append((x, y, w, h))

    if len(proposals) == 0:
        continue

    # Batch all image regions
    image_batch = torch.cat(proposals).to(device)

    # Encode all at once
    with torch.no_grad():
        image_features = model.encode_image(image_batch)
        similarity_scores = (image_features @ text_features.T).softmax(dim=-1)

    detected_objects = []
    for idx, scores in enumerate(similarity_scores):
        max_idx = torch.argmax(scores).item()
        confidence = scores[max_idx].item()

        if confidence > 0.85:
            x, y, w, h = coordinates[idx]
            label = object_descriptions[max_idx]
            detected_objects.append((x, y, w, h, label, confidence))

    # Draw bounding boxes
    for (x, y, w, h, obj, conf) in detected_objects:
        label = f"{obj} ({conf:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Optimized CLIP Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
