from flask import Flask

from flask import jsonify, request

from hashlib import algorithms_available
import tensorflow
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
from keras.layers import GlobalMaxPooling2D
import numpy as np
from numpy.linalg import norm
import math
from io import BytesIO
import pickle
import base64
from PIL import Image
from sklearn.neighbors import NearestNeighbors


import pyrebase


firebaseConfig = {
    "apiKey": "AIzaSyAxYCMEyxW4WS5AX2WEGantpgJJkPwbNtI",
    "authDomain": "test-pbl6.firebaseapp.com",
    "projectId": "test-pbl6",
    "storageBucket": "test-pbl6.appspot.com",
    "messagingSenderId": "713437593650",
    "appId": "1:713437593650:web:ed2ab6399b941b183b3c2b",
    "databaseURL": ""
}
firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()

def download_featurefile():
    path_on_clould = "feature/featurefilenameRes152.pk1"
    path_local =     "static\\featurefilenameRes152.pk1"
    storage.child(path_on_clould).download(
       path_local)
    
    path_name_cloud = "feature/featurevectorRes152.pk1"
    path_name_local = "static\\featurevectorRes152.pk1"
    storage.child(path_name_cloud).download(
       path_name_local)
    

def upload_featurefile():
    
    # path_on_clould = "image.jpg"
    # path_local = "c:\\Users\\ASUS\\Desktop\\DoAN7\\SimilaritySearch\\FootwearDataset\\FootwearImg\\56994.jpg"
    # storage.child(path_on_clould).put(
    #    path_local)
    path_on_clould = "feature/featurefilenameRes152.pk1"
    path_local = "static\\featurefilenameRes152.pk1"
    storage.child(path_on_clould).put(
       path_local)
    
    path_name_cloud = "feature/featurevectorRes152.pk1"
    path_name_local = "static\\featurevectorRes152.pk1"
    storage.child(path_name_cloud).put(
       path_name_local)
    


app = Flask(__name__)  
    
    

featute_list = np.array(pickle.load(
    open("static\\featurevectorRes152.pk1", "rb")))
filenames = pickle.load(open(
    "static\\featurefilenameRes152.pk1", "rb"))




model = ResNet152(weights='imagenet', include_top=False,
                  input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
model. summary()


@app.route("/", methods=['POST'])
def extract_feature():
    
    encode = request.json['namefoto']
    
    img = Image.open(BytesIO(base64.b64decode(encode)))
    # img = cv2.imread("FootwearDataset/FootwearImg/59050.jpg")
    img = img.resize((224, 224))
    
    

    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result/norm(result)

    neighbors = NearestNeighbors(
        n_neighbors=16, algorithm="brute", metric="euclidean")
    neighbors.fit(featute_list)

    distance, indices = neighbors.kneighbors([normalized])
    print(indices)
    print("dis")
    print(distance)

    K = 16
    name = []
    for id in range(K):
        str = filenames[indices[0][id]]
        arr1 = str.split("\\")[-1]
        name.append(arr1)
    return jsonify(name)



    
    # encode = request.json['namefoto']
    
    # img = Image.open(BytesIO(base64.b64decode(encode)))
    # # img = cv2.imread("FootwearDataset/FootwearImg/59050.jpg")
    # img = img.resize((224, 224))
    
    
    
    
    # img = np.array(img)
    # expand_img = np.expand_dims(img, axis=0)
    # pre_img = preprocess_input(expand_img)
    # result = model.predict(pre_img).flatten()
    # normalized = result/norm(result)

    # neighbors = NearestNeighbors(
    #     n_neighbors=16, algorithm="brute", metric="euclidean")
    # neighbors.fit(featute_list)

    # distance, indices = neighbors.kneighbors([normalized])
    # print(indices)
    # print("dis")
    # print(distance)

    # K = 16
    # name = []
    # for id in range(K):
    #     str = filenames[indices[0][id]]
    #     arr1 = str.split("\\")[-1]
    #     name.append(arr1)
    # return jsonify(name)



if __name__ == '__main__':
    app.run( port=5000)