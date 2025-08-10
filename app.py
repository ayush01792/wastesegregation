import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition
class WasteClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(WasteClassifier, self).__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Load Model
model = WasteClassifier(num_classes=len(CLASSES)).to(device)
try:
    state_dict = torch.load('model/best_model.pth', map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("✅ Model loaded")
except Exception as e:
    print("⚠️ Failed to load model:", e)

model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_idx = torch.argmax(probs).item()
    return CLASSES[predicted_idx], probs.cpu().numpy()

# Recycling instructions helper
def get_recycling_instructions(class_name):
    instructions = {
        'plastic': 'Rinse and place in the recycling bin.',
        'paper': 'Flatten and keep dry before recycling.',
        'glass': 'Rinse and remove caps.',
        'metal': 'Crush cans to save space.',
        'cardboard': 'Flatten before placing in the bin.',
        'trash': 'Dispose of in general waste.'
    }
    return instructions.get(class_name.lower(), 'No instructions available.')

# Make helper function available in templates
@app.context_processor
def inject_utilities():
    return dict(get_recycling_instructions=get_recycling_instructions)

# Main route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('index.html', error="No file selected")
        if not allowed_file(file.filename):
            return render_template('index.html', error="Invalid file type")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            class_name, probabilities = predict_image(filepath)
            confidence = max(probabilities)
            results = {
                'image_path': filename,
                'class_name': class_name,
                'confidence': confidence,
                'probabilities': dict(zip(CLASSES, probabilities))
            }
            return render_template('result.html', results=results)
        except Exception as e:
            print("Prediction error:", e)
            return render_template('index.html', error="Prediction failed")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
