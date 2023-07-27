from flask import Flask, render_template, request, session, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)
app.secret_key = 'lmnop23'

# load the model
model_path = '../dogsncats/model_weights/model_weights_epoch49.pt'

model = nn.Sequential()
model.add_module(
    'conv1',
    nn.Conv2d(in_channels=3, out_channels=32,
              kernel_size=5, padding=2)
)
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout1', nn.Dropout(p=0.2))
model.add_module(
    'conv2',
    nn.Conv2d(in_channels=32, out_channels=64,
              kernel_size=5, padding=2)
)
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout2', nn.Dropout(p=0.2))
model.add_module('flatten', nn.Flatten())
model.add_module('fc1', nn.Linear(196608, 1024))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout3', nn.Dropout(p=0.5))
model.add_module('fc2', nn.Linear(1024, 1))


model.load_state_dict(torch.load(model_path))
model.eval()

# define a transformer for the uploaded image
img_width, img_height = 256, 192
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor()])

## path to save the files to and retreive files from
upload_path = './static/uploads/'

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return 'No file uploaded', 400

        file = request.files['file']

        # Check if the file is selected
        if file.filename == '':
            return 'No file selected', 400
        
        if not file.filename.endswith('.jpg'):
            return jsonify({'status': 'error', 'message': 'Only JPG images are allowed.'})

        # Save the file to a desired location (e.g., 'uploads' folder)
        img_src = upload_path + file.filename
        file.save(img_src)
        
        # save this for use in /classify
        session['img_src'] = img_src

        # Render the template with the uploaded image
        return render_template('index.html', filename=file.filename)

    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    
    img_src = session.get('img_src')
    
    # Load the image from the source
    image = Image.open(img_src)
    image = transform(image).unsqueeze(0)

    # Pass the image through the model for prediction
    with torch.no_grad():
        outputs = model(image)
        
        sigmoid = nn.Sigmoid()
        probability = sigmoid(outputs)
        
        threshold = 0.5
        prediction = probability >= threshold
        
        if prediction.item():
            prediction = 'dog'
        else:
            prediction = 'cat'

    # Return the predicted label as a JSON response
    response = {'predicted_label': prediction}
    return jsonify(response)


if __name__ == '__main__':
    app.debug = True  # Enable debug mode
    app.run()
