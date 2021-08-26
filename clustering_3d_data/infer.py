import torch
import argparse
import os
from model.model import PCAutoEncoder
from model.model_fxia22 import PointNetAE
import open3d as o3d
import numpy as np
from sklearn.externals import joblib 
from sklearn import cluster
from util import pointutil

parser = argparse.ArgumentParser()

parser.add_argument("--input_folder", required=True, help="Single 3d model or input folder containing 3d models")
parser.add_argument("--nn_model", required=True, help="Trained Neural Network Model")
parser.add_argument("--nn_model_type", required=True, choices=['fxia', 'dhiraj'], help="Model Type")
parser.add_argument("--out_norm_input", action="store_true", help="Output normalized version of input file")
parser.add_argument("--classifier_model", required=True, help="Path to the Classifier Model")

ip_options = parser.parse_args()
input_folder = ip_options.input_folder

# Setting the Gradient Calculation Feature off
# torch.set_grad_enabled(False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

point_dim = 3
num_points = 2048

# Load Autoencoder Model
if ip_options.nn_model_type == 'dhiraj':
    autoencoder = PCAutoEncoder(point_dim, num_points)
elif ip_options.nn_model_type == 'fxia':
    autoencoder = PointNetAE(num_points)

state_dict = torch.load(ip_options.nn_model, map_location=device)
autoencoder.load_state_dict(state_dict)

# Load Classifer Model
kmeans = joblib.load(ip_options.classifier_model)

def save_as_pcd(iFileName, iPoints, iColor=None):
    # Create pcd file of the reconstructed points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(iPoints)
    if iColor is not None:
        pcd.colors = o3d.utility.Vector3dVector(iColor)
    #outPcdFile = os.path.join(os.path.dirname(iFileName), os.path.splitext(os.path.basename(iFileName))[0] + "_out.pcd")
    o3d.io.write_point_cloud(iFileName, pcd, write_ascii=True)

def get_path_without_ext(iFilePath):
    oFilePath = os.path.join(os.path.dirname(iFilePath), os.path.splitext(os.path.basename(iFilePath))[0])
    return oFilePath


def generate_color(iPoints, iColor):
    color_matrix = np.full(iPoints.shape, iColor)
    return color_matrix

def infer_model_file(input_file, autoencoder):
    
    print(f"Processing file: {input_file}")
    cloud = o3d.io.read_point_cloud(input_file)
    points = np.array(cloud.points)

    # extract only "N" number of point from the Point Cloud
    points = pointutil.random_n_points(points, num_points)

    # Normalize and center and bring it to unit sphere
    points = pointutil.normalize(points)

    if ip_options.out_norm_input:
        norm_ip_file = get_path_without_ext(input_file) + "_norm.pcd"
        save_as_pcd(norm_ip_file, points)

    points = torch.from_numpy(points).float()
    points = torch.unsqueeze(points, 0) #done to introduce batch_size of 1 
    points = points.transpose(2, 1)
    # points = points.cuda() #uncomment this if running on GPU
    autoencoder = autoencoder.eval()
    reconstructed_points, latent_vector = autoencoder(points)
    
    # classify
    cluster_id = kmeans.predict(latent_vector)
    print(f"Predicted Cluster ID : {cluster_id}")

    #Reshape 
    reconstructed_points = reconstructed_points.squeeze().transpose(0,1)
    reconstructed_points = reconstructed_points.numpy()
    
    outPcdFile = get_path_without_ext(input_file) + "_out.pcd"
    green_color = (0, 255, 0)
    color = generate_color(reconstructed_points, green_color)
    save_as_pcd(outPcdFile, reconstructed_points, color)



def infer_models_folder(input_folder, autoencoder):
    for root, subdirs, files in os.walk(ip_options.input_folder):
        for fileName in files:
            ipFilePath = os.path.join(root, fileName)
            # check the file 
            if ipFilePath.endswith('.pcd'):
                infer_model_file(ipFilePath, autoencoder)
       

with torch.no_grad():
    if os.path.isdir(input_folder):
        infer_models_folder(input_folder, autoencoder)
    elif os.path.isfile(input_folder):
        infer_model_file(input_folder, autoencoder)

