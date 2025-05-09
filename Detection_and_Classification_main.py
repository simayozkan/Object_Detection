from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt

#image classification using visual transformation
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

image = plt.imread('laptop.jpeg')
plt.imshow(image)

inputs = feature_extractor(images=image, return_tensors="pt")

pixel_values = inputs["pixel_values"]

outputs = model(pixel_values)

logits = outputs.logits
logits.shape

predicted_class_idx = logits.argmax(-1).item()
predicted_class_idx

predicted_class = model.config.id2label[predicted_class_idx]
predicted_class 

#zero_shot image classification
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

checkpoint = "openai/clip-vit-large-patch14"

model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint)

processor = AutoProcessor.from_pretrained(checkpoint)



url = ""

image = Image.open(requests.get(url, stream=True).raw)

image

candidate_labels = ["tree", "car", "bike", "cat"]

inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)



import torch

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits_per_image[0]
probs = logits.softmax(dim=-1).numpy()

scores = probs.tolist()

result = [
    {"score": score, "label": candidate_label}
    for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
]

result

#zero shot object detection
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
  
checkpoint = "google/owlvit-base-patch32"

model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)

processor = AutoProcessor.from_pretrained(checkpoint)

url = ""

im = Image.open(requests.get(url, stream=True).raw)

im

text_queries = ["hat", "book", "sunglasses", "camera"]

inputs = processor(text=text_queries, images=im, return_tensors="pt")

from PIL import ImageDraw

with torch.no_grad():
    outputs = model(**inputs)
    target_sizes = torch.tensor([im.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

draw = ImageDraw.Draw(im)

scores = results["scores"].tolist()
labels = results["labels"].tolist()
boxes = results["boxes"].tolist()


for box, score, label in zip(boxes, scores, labels):
    xmin, ymin, xmax, ymax = box
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{text_queries[label]}: {round(score,2)}", fill="white")

im
