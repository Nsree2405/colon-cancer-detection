import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from transformers import CLIPProcessor, CLIPModel

# ==========================================================
# DEVICE
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# LOAD CLIP MODEL (STRICT FILTER)
# ==========================================================
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def is_medical_image(image):
    image = image.convert("RGB")

    texts = [
        "a histopathology image",
        "microscopy image of tissue",
        "medical slide image",
        "colon biopsy image",
        "human face",
        "portrait photo",
        "animal photo",
        "car image",
        "everyday object"
    ]

    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    medical_scores = probs[:4]
    non_medical_scores = probs[4:]

    max_medical = medical_scores.max()
    max_non_medical = non_medical_scores.max()

    # 🔥 STRICT RULES
    if max_medical < 0.30:
        return False

    if max_medical < max_non_medical + 0.10:
        return False

    return True

# ==========================================================
# MODEL
# ==========================================================
class ColonModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_v2_s(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(features)

# ==========================================================
# LOAD MODELS
# ==========================================================
binary_model = ColonModel(2)
binary_model.load_state_dict(torch.load("best_model_binary.pth", map_location=device)["model_state_dict"])
binary_model.to(device).eval()

binary_classes = ["Normal", "Tumor"]

multiclass_classes = ["ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM"]
multiclass_model = ColonModel(len(multiclass_classes))
multiclass_model.load_state_dict(torch.load("best_multiclass_model.pth", map_location=device)["model_state_dict"])
multiclass_model.to(device).eval()

severity_classes = ["Grade 1","Grade 2","Grade 3","Normal"]
severity_model = efficientnet_v2_s(weights=None)
in_features = severity_model.classifier[1].in_features
severity_model.classifier[1] = nn.Linear(in_features, len(severity_classes))
severity_model.load_state_dict(torch.load("best_model_severity.pth", map_location=device))
severity_model.to(device).eval()

# ==========================================================
# TRANSFORM
# ==========================================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ==========================================================
# SAFE PREDICT (🔥 FINAL FIX)
# ==========================================================
def safe_predict(model, image, classes):
    image = image.convert("RGB")

    # 🚨 STEP 1: CLIP FILTER
    if not is_medical_image(image):
        return "❌ This is NOT a colon histopathology / medical image."

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    max_prob = probs.max()

    # 🚨 STEP 2: CONFIDENCE FILTER
    if max_prob < 0.70:
        return "⚠️ Uncertain prediction — input may not be a valid colon tissue image."

    # ✅ NORMAL OUTPUT
    result = ""
    for i, cls in enumerate(classes):
        result += f"{cls}: {probs[i]*100:.2f}%\n"

    final_pred = classes[probs.argmax()]
    result += f"\nFinal Prediction: {final_pred}"

    return result

# ==========================================================
# FUNCTIONS
# ==========================================================
def predict_binary(img):
    return safe_predict(binary_model, img, binary_classes)

def predict_multiclass(img):
    return safe_predict(multiclass_model, img, multiclass_classes)

def predict_severity(img):
    return safe_predict(severity_model, img, severity_classes)

# ==========================================================
# UI
# ==========================================================
with gr.Blocks(title="Colon Cancer AI System") as demo:

    gr.Markdown("""
    # 🧬 Colon Cancer AI System
    
    - Binary Classification  
    - Multi-Class Classification  
    - Severity Grading  
    
    ⚠️ Non-medical images are automatically rejected.
    """)

    with gr.Tab("🔍 Binary"):
        i1 = gr.Image(type="pil")
        o1 = gr.Textbox()
        gr.Button("Predict").click(predict_binary, i1, o1)

    with gr.Tab("🧪 Multi-Class"):
        i2 = gr.Image(type="pil")
        o2 = gr.Textbox()
        gr.Button("Predict").click(predict_multiclass, i2, o2)

    with gr.Tab("📊 Severity"):
        i3 = gr.Image(type="pil")
        o3 = gr.Textbox()
        gr.Button("Predict").click(predict_severity, i3, o3)

demo.launch()


# 50% correect version
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import efficientnet_v2_s
# import torchvision.transforms as transforms
# from PIL import Image
# import gradio as gr
# from transformers import CLIPProcessor, CLIPModel

# # ==========================================================
# # DEVICE
# # ==========================================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ==========================================================
# # LOAD CLIP MODEL (🔥 NEW)
# # ==========================================================
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# def is_medical_image(image):
#     image = image.convert("RGB")

#     texts = [
#         "a histopathology image of colon tissue",
#         "a microscopy image of tissue",
#         "a medical slide image",
#         "a photo of a cat",
#         "a photo of a car",
#         "a normal everyday object"
#     ]

#     inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)

#     with torch.no_grad():
#         outputs = clip_model(**inputs)
#         probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

#     # First 3 = medical, last 3 = non-medical
#     medical_score = probs[:3].max()
#     non_medical_score = probs[3:].max()

#     if medical_score > non_medical_score:
#         return True
#     return False

# # ==========================================================
# # YOUR MODEL
# # ==========================================================
# class ColonModel(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.backbone = efficientnet_v2_s(weights=None)
#         num_features = self.backbone.classifier[1].in_features
#         self.backbone.classifier = nn.Identity()

#         self.classifier = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         features = self.backbone(x)
#         if features.dim() > 2:
#             features = F.adaptive_avg_pool2d(features, 1).flatten(1)
#         return self.classifier(features)

# # ==========================================================
# # LOAD MODELS
# # ==========================================================
# binary_model = ColonModel(2)
# binary_model.load_state_dict(torch.load("best_model_binary.pth", map_location=device)["model_state_dict"])
# binary_model.to(device).eval()

# multiclass_classes = ["ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM"]
# multiclass_model = ColonModel(len(multiclass_classes))
# multiclass_model.load_state_dict(torch.load("best_multiclass_model.pth", map_location=device)["model_state_dict"])
# multiclass_model.to(device).eval()

# severity_classes = ["Grade 1","Grade 2","Grade 3","Normal"]
# severity_model = efficientnet_v2_s(weights=None)
# in_features = severity_model.classifier[1].in_features
# severity_model.classifier[1] = nn.Linear(in_features, len(severity_classes))
# severity_model.load_state_dict(torch.load("best_model_severity.pth", map_location=device))
# severity_model.to(device).eval()

# binary_classes = ["Normal","Tumor"]

# # ==========================================================
# # TRANSFORM
# # ==========================================================
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# # ==========================================================
# # SAFE PREDICT
# # ==========================================================
# def safe_predict(model, image, classes):
#     image = image.convert("RGB")

#     # 🚨 CLIP CHECK
#     if not is_medical_image(image):
#         return "❌ This is NOT a histopathology / colon image."

#     img = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(img)
#         probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

#     result = ""
#     for i, cls in enumerate(classes):
#         result += f"{cls}: {probs[i]*100:.2f}%\n"

#     final_pred = classes[probs.argmax()]
#     result += f"\nFinal Prediction: {final_pred}"

#     return result

# # ==========================================================
# # FUNCTIONS
# # ==========================================================
# def predict_binary(img):
#     return safe_predict(binary_model, img, binary_classes)

# def predict_multiclass(img):
#     return safe_predict(multiclass_model, img, multiclass_classes)

# def predict_severity(img):
#     return safe_predict(severity_model, img, severity_classes)

# # ==========================================================
# # UI
# # ==========================================================
# with gr.Blocks(title="Colon Cancer AI System") as demo:

#     gr.Markdown("# 🧬 Colon Cancer AI System")

#     with gr.Tab("Binary"):
#         i1 = gr.Image(type="pil")
#         o1 = gr.Textbox()
#         gr.Button("Predict").click(predict_binary, i1, o1)

#     with gr.Tab("Multi-Class"):
#         i2 = gr.Image(type="pil")
#         o2 = gr.Textbox()
#         gr.Button("Predict").click(predict_multiclass, i2, o2)

#     with gr.Tab("Severity"):
#         i3 = gr.Image(type="pil")
#         o3 = gr.Textbox()
#         gr.Button("Predict").click(predict_severity, i3, o3)

# demo.launch()






# v2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import efficientnet_v2_s, resnet18
# import torchvision.transforms as transforms
# from PIL import Image
# import gradio as gr

# # ==========================================================
# # DEVICE
# # ==========================================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ==========================================================
# # LOAD IMAGENET LABELS
# # ==========================================================
# import urllib.request

# LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# urllib.request.urlretrieve(LABELS_URL, "imagenet_classes.txt")

# with open("imagenet_classes.txt") as f:
#     imagenet_classes = [line.strip() for line in f.readlines()]

# # ==========================================================
# # GATE MODEL (NEW)
# # ==========================================================
# gate_model = resnet18(weights="DEFAULT").to(device)
# gate_model.eval()

# gate_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# # Keywords that indicate NON-medical images
# NON_MEDICAL_KEYWORDS = [
#     "cat","dog","car","person","bird","truck","chair","table",
#     "food","animal","vehicle","flower","tree","sky","building"
# ]

# def is_medical_image(image):
#     image = image.convert("RGB")
#     img = gate_transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = gate_model(img)
#         probs = F.softmax(outputs, dim=1)
#         top5 = torch.topk(probs, 5)

#     labels = [imagenet_classes[i] for i in top5.indices[0]]

#     # 🚨 reject if clearly non-medical
#     for label in labels:
#         for keyword in NON_MEDICAL_KEYWORDS:
#             if keyword in label.lower():
#                 return False

#     return True

# # ==========================================================
# # YOUR MODEL
# # ==========================================================
# class ColonModel(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.backbone = efficientnet_v2_s(weights=None)
#         num_features = self.backbone.classifier[1].in_features
#         self.backbone.classifier = nn.Identity()

#         self.classifier = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         features = self.backbone(x)
#         if features.dim() > 2:
#             features = F.adaptive_avg_pool2d(features, 1).flatten(1)
#         return self.classifier(features)

# # ==========================================================
# # LOAD YOUR MODELS
# # ==========================================================
# binary_model = ColonModel(2)
# binary_model.load_state_dict(torch.load("best_model_binary.pth", map_location=device)["model_state_dict"])
# binary_model.to(device).eval()

# multiclass_classes = ["ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM"]
# multiclass_model = ColonModel(len(multiclass_classes))
# multiclass_model.load_state_dict(torch.load("best_multiclass_model.pth", map_location=device)["model_state_dict"])
# multiclass_model.to(device).eval()

# severity_classes = ["Grade 1","Grade 2","Grade 3","Normal"]
# severity_model = efficientnet_v2_s(weights=None)
# in_features = severity_model.classifier[1].in_features
# severity_model.classifier[1] = nn.Linear(in_features, len(severity_classes))
# severity_model.load_state_dict(torch.load("best_model_severity.pth", map_location=device))
# severity_model.to(device).eval()

# binary_classes = ["Normal","Tumor"]

# # ==========================================================
# # TRANSFORM
# # ==========================================================
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# # ==========================================================
# # SAFE PREDICT (FINAL)
# # ==========================================================
# def safe_predict(model, image, classes):
#     image = image.convert("RGB")

#     # 🚨 GATE CHECK
#     if not is_medical_image(image):
#         return "⚠️ Uploaded image is not a histopathological / medical image."

#     img = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(img)
#         probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

#     result = ""
#     for i, cls in enumerate(classes):
#         result += f"{cls}: {probs[i]*100:.2f}%\n"

#     final_pred = classes[probs.argmax()]
#     result += f"\nFinal Prediction: {final_pred}"

#     return result

# # ==========================================================
# # FUNCTIONS
# # ==========================================================
# def predict_binary(img):
#     return safe_predict(binary_model, img, binary_classes)

# def predict_multiclass(img):
#     return safe_predict(multiclass_model, img, multiclass_classes)

# def predict_severity(img):
#     return safe_predict(severity_model, img, severity_classes)

# # ==========================================================
# # UI
# # ==========================================================
# with gr.Blocks(title="Colon Cancer AI System") as demo:

#     gr.Markdown("# 🧬 Colon Cancer AI System")

#     with gr.Tab("Binary"):
#         i1 = gr.Image(type="pil")
#         o1 = gr.Textbox()
#         gr.Button("Predict").click(predict_binary, i1, o1)

#     with gr.Tab("Multi-Class"):
#         i2 = gr.Image(type="pil")
#         o2 = gr.Textbox()
#         gr.Button("Predict").click(predict_multiclass, i2, o2)

#     with gr.Tab("Severity"):
#         i3 = gr.Image(type="pil")
#         o3 = gr.Textbox()
#         gr.Button("Predict").click(predict_severity, i3, o3)

# demo.launch()
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import efficientnet_v2_s
# import torchvision.transforms as transforms
# from PIL import Image
# import gradio as gr

# # ==========================================================
# # DEVICE
# # ==========================================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ==========================================================
# # SHARED MODEL ARCHITECTURE
# # ==========================================================

# class ColonModel(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.backbone = efficientnet_v2_s(weights=None)
#         num_features = self.backbone.classifier[1].in_features
#         self.backbone.classifier = nn.Identity()

#         self.classifier = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         features = self.backbone(x)
#         if features.dim() > 2:
#             features = F.adaptive_avg_pool2d(features, 1).flatten(1)
#         return self.classifier(features)

# # ==========================================================
# # LOAD BINARY MODEL
# # ==========================================================

# binary_model = ColonModel(num_classes=2)
# binary_checkpoint = torch.load("best_model_binary.pth", map_location=device)
# binary_model.load_state_dict(binary_checkpoint["model_state_dict"])
# binary_model.to(device)
# binary_model.eval()

# binary_classes = ["Normal", "Tumor"]

# # ==========================================================
# # LOAD MULTI-CLASS MODEL
# # ==========================================================

# multiclass_classes = [
#     "ADI", "BACK", "DEB", "LYM",
#     "MUC", "MUS", "NORM", "STR", "TUM"
# ]

# multiclass_model = ColonModel(num_classes=len(multiclass_classes))
# multi_checkpoint = torch.load("best_multiclass_model.pth", map_location=device)
# multiclass_model.load_state_dict(multi_checkpoint["model_state_dict"])
# multiclass_model.to(device)
# multiclass_model.eval()

# # ==========================================================
# # LOAD SEVERITY MODEL (4 CLASSES)
# # ==========================================================

# # ⚠️ Make sure this order matches your training dataset (alphabetical likely)
# severity_classes = [
#     "Grade 1",
#     "Grade 2",
#     "Grade 3",
#     "Normal"
# ]

# # ==========================================================
# # LOAD SEVERITY MODEL (CORRECT VERSION)
# # ==========================================================

# from torchvision.models import efficientnet_v2_s

# severity_classes = ["Grade 1", "Grade 2", "Grade 3", "Normal"]  # confirm order

# severity_model = efficientnet_v2_s(weights=None)

# # Replace classifier exactly like training
# in_features = severity_model.classifier[1].in_features
# severity_model.classifier[1] = nn.Linear(in_features, len(severity_classes))

# # Load weights (direct state_dict)
# severity_model.load_state_dict(torch.load("best_model_severity.pth", map_location=device))

# severity_model.to(device)
# severity_model.eval()

# # ==========================================================
# # TRANSFORM (MATCH TRAINING)
# # ==========================================================

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
# ])

# # ==========================================================
# # PREDICTION FUNCTIONS
# # ==========================================================

# def predict_binary(image):
#     image = image.convert("RGB")
#     image = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = binary_model(image)
#         probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

#     result = ""
#     for i, cls in enumerate(binary_classes):
#         result += f"{cls}: {probs[i]*100:.2f}%\n"

#     final_pred = binary_classes[probs.argmax()]
#     result += f"\nFinal Prediction: {final_pred}"

#     return result


# def predict_multiclass(image):
#     image = image.convert("RGB")
#     image = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = multiclass_model(image)
#         probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

#     result = ""
#     for i, cls in enumerate(multiclass_classes):
#         result += f"{cls}: {probs[i]*100:.2f}%\n"

#     final_pred = multiclass_classes[probs.argmax()]
#     result += f"\nFinal Prediction: {final_pred}"

#     return result


# def predict_severity(image):
#     image = image.convert("RGB")
#     image = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = severity_model(image)
#         probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

#     result = ""
#     for i, cls in enumerate(severity_classes):
#         result += f"{cls}: {probs[i]*100:.2f}%\n"

#     final_pred = severity_classes[probs.argmax()]
#     result += f"\nFinal Prediction: {final_pred}"

#     return result

# # ==========================================================
# # UI (GRADIO TABS)
# # ==========================================================

# with gr.Blocks(title="Colon Cancer AI System") as demo:

#     gr.Markdown("""
#     # 🧬 Colon Cancer AI System
    
#     ### Features:
#     - 🔍 Binary Detection (Normal vs Tumor)
#     - 🧪 Multi-Class Classification (9 tissue types)
#     - 📊 Severity Grading (Normal + Grade 1/2/3)

#     ⚠️ **Disclaimer:** This system is for research purposes only and not for clinical diagnosis.
#     """)

#     # ---------------------------
#     # TAB 1 — BINARY
#     # ---------------------------
#     with gr.Tab("🔍 Binary Detection"):
#         gr.Markdown("Upload image to detect **Normal vs Tumor**")
#         img1 = gr.Image(type="pil")
#         out1 = gr.Textbox(label="Prediction")
#         btn1 = gr.Button("Predict")
#         btn1.click(predict_binary, inputs=img1, outputs=out1)

#     # ---------------------------
#     # TAB 2 — MULTI-CLASS
#     # ---------------------------
#     with gr.Tab("🧪 Multi-Class Classification"):
#         gr.Markdown("Classify into detailed tissue categories")
#         img2 = gr.Image(type="pil")
#         out2 = gr.Textbox(label="Prediction")
#         btn2 = gr.Button("Predict")
#         btn2.click(predict_multiclass, inputs=img2, outputs=out2)

#     # ---------------------------
#     # TAB 3 — SEVERITY
#     # ---------------------------
#     with gr.Tab("📊 Severity Grading"):
#         gr.Markdown("Predict cancer severity level")
#         img3 = gr.Image(type="pil")
#         out3 = gr.Textbox(label="Prediction")
#         btn3 = gr.Button("Predict")
#         btn3.click(predict_severity, inputs=img3, outputs=out3)

# demo.launch()