from flask import Flask

from flask import jsonify, request
import numpy as np
from numpy.linalg import norm
# import math
from io import BytesIO
# import pickle
import base64
from PIL import Image
# from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from  rembg  import remove
from tensorflow.keras.models import load_model

model_load =   load_model("static/trainResnet152.h5")
# model_load =   load_model("static/trainResnet50-1000Image1.h5")
class_names = ['Boots - Ankle',
 'Boots - Knee High',
 'Flip flops',
 'Sandals - Flat',
 'Shoes - Flats',
 'Shoes - Heels',
 'Shoes - Oxfords',
 'Sneaker & Athletic']
# import matplotlib.pyplot as plt




app = Flask(__name__)  
    
    
@app.route("/ai", methods=['GET'])    
def test():
    name = []
    name.append("sads")
    name.append("dsds")
    return jsonify(name)    
    



@app.route("/ai/api/clf", methods=['POST'])
def extract_feature():
    
    
    encode = request.json['imageBase64']
    img = Image.open(BytesIO(base64.b64decode(encode)))

    png = remove(img)
    background = Image.new('RGBA', png.size, (255, 255, 255))
    img = (Image.alpha_composite(background, png)).convert("RGB")
    
    
    img = img.resize((224, 224))
    img = np.array(img)
    pre_img = np.expand_dims(img, axis=0)
    
    
    result = model_load.predict(pre_img)
    output_class= class_names[np.argmax(result)]
    print("The predicted class is", output_class)


    return   jsonify(output_class)



if __name__ == '__main__':
    app.run( host= '0.0.0.0')