import cv2
import numpy as np
import open3d as o3d
import argparse


left_file = '/home/meryl/cv/ps7/ps7-images/jmeyerso-left.jpeg'
left_color = cv2.imread(left_file)
left_color = cv2.cvtColor(left_color, cv2.COLOR_BGR2RGB)
left = cv2.imread(left_file, cv2.IMREAD_GRAYSCALE) # left image, convert to grayscale

right_file = '/home/meryl/cv/ps7/ps7-images/jmeyerso-right.jpeg'
right = cv2.imread(right_file, cv2.IMREAD_GRAYSCALE) # right image, convert to grayscale

stereo = cv2.StereoSGBM_create(numDisparities=16*6, blockSize=15) 
    #how pixels shifted between left and right images
    #numDisparities = range of pixel shifts
    #blockSize = size of square patch for comparison

disp_img = stereo.compute(left, right) #computes 

cv2.namedWindow("Image",cv2.WINDOW_NORMAL) #window
cv2.imshow("Image", disp_img.astype(np.uint8)) #show image

cv2.imwrite("jmeyerso-disparity.jpeg", disp_img.astype(np.uint8)) #save image

#delt Z = f /  b/ (U - U') 
#map 2d pixel coords & disparity values into 3D space using camera calibration parameters

disp = disp_img.astype(np.float32) / 16.0 #convert back

h, w = left.shape
cx = w // 2 #optical center
cy = h // 2 #optical center
f = w // 2 #focal length estimate; will fix later
Tx = 0.1 * w #baseline translation estimate

Q = np.float64([[1, 0, 0, -cx], 
             [0, 1, 0, -cy], 
             [0, 0, 0, f], 
             [0, 0, -1/Tx, 0]]) #camera parameter matrix

#Reproject 2D single-channel disparity image into 3-channel point cloud with 4x4 perspective transform
point_cloud = cv2.reprojectImageTo3D(disp, Q).reshape(-1, 3) #(n, 3) array of XYZ coords

#Filtering
min = 20 #min disparity --> background
max = 80 #max disparity --> too close

mask = (disp.reshape(-1) > min) & (disp.reshape(-1) < max) & (np.all(np.isfinite(point_cloud), axis = 1))
points = point_cloud[mask]

#Color point cloud
colors = left_color.reshape(-1, 3) #(H*W, 3)
colors = colors[mask].astype(np.float64) / 255.0 #Open 3d wants colors in 1,2

#Point cloud object: sets points and colors
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)
cloud.colors = o3d.utility.Vector3dVector(colors)

#file writing and visualization
o3d.io.write_point_cloud("jmeyerso.ply", cloud, write_ascii=False)
ply = o3d.io.read_point_cloud("jmeyerso.ply")
o3d.visualization.draw_geometries([ply])

scale = 100

cv2.waitKey()