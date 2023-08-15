import io
import os
from PIL import Image
from sklearn.cluster import DBSCAN
from imutils import build_montages, paths
import numpy as np
import os
import pickle
import cv2
import shutil
import time
#import dlib $TODO
from pyPiper import Node, Pipeline
from tqdm import tqdm
from moviepy.editor import *
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from google.cloud import videointelligence_v1 as videointelligence

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\USER\\AppData\\Roaming\\gcloud\\application_default_credentials.json"

''' Common utilities '''
'''
Credits: AndyP at StackOverflow
The ResizeUtils provides resizing function to keep the aspect ratio intact
'''
class ResizeUtils:
    # Given a target height, adjust the image by calculating the width and resize
    def rescale_by_height(self, image, target_height, method=cv2.INTER_LANCZOS4):
        """Rescale `image` to `target_height` (preserving aspect ratio)."""
        w = int(round(target_height * image.shape[1] / image.shape[0]))
        return cv2.resize(image, (w, target_height), interpolation=method)

    # Given a target width, adjust the image by calculating the height and resize
    def rescale_by_width(self, image, target_width, method=cv2.INTER_LANCZOS4):
        """Rescale `image` to `target_width` (preserving aspect ratio)."""
        h = int(round(target_width * image.shape[0] / image.shape[1]))
        return cv2.resize(image, (target_width, h), interpolation=method)

def detect_faces(gcs_uri):
    """Detects faces in a video."""
    
    client = videointelligence.VideoIntelligenceServiceClient()

    # Configure the request
    config = videointelligence.FaceDetectionConfig(
        include_bounding_boxes=True, include_attributes=True
    )
    context = videointelligence.VideoContext(face_detection_config=config)

    # Start the asynchronous request
    operation = client.annotate_video(
        request={
            "features": [videointelligence.Feature.FACE_DETECTION],
            "input_uri": gcs_uri,
            "video_context": context,
        }
    )

    print("\nProcessing video for face detection annotations.")
    result = operation.result(timeout=10000)

    print("\nFinished processing.\n")

    # Retrieve the first result, because a single video was processed.
    return result.annotation_results[0] #TODO - few faces

def extract_faces(annotation_result,movie_name,gcs_uri,output):
    """ Extract faces pictures to a folder """
    if os.path.exists(output):
        shutil.rmtree(output)
        time.sleep(0.5)
    os.mkdir(output)
    faces_dir = os.path.join(output,"Faces")
    os.mkdir(faces_dir)
    
    # TODO movie = UPLOAD FROM GOOGLE CLOUD
    movie = VideoFileClip(str("D:\\OneDrive\\OneDrive - mail.tau.ac.il\\python\\face_detection\\" + movie_name + ".mp4"))
    i = 0
    
    for annotation in annotation_result.face_detection_annotations:
        for track in annotation.tracks:
            if track.confidence > 0.6:
                i+=1
                #print("Face detected:")
                #print(
                #    "Segment: {}s to {}s".format(
                #        track.segment.start_time_offset.seconds
                #        + track.segment.start_time_offset.microseconds / 1e6,
                #        track.segment.end_time_offset.seconds
                #        + track.segment.end_time_offset.microseconds / 1e6,
                #    )
                #)

                new_image_path = os.path.join(output,"image"+str(i)+"full.jpg")
                #print("image " + str(i) + " time_offset: " + str(track.timestamped_objects[0].time_offset)
                #      + ", in seconds: " + str(track.timestamped_objects[0].time_offset.seconds + track.timestamped_objects[0].time_offset.microseconds / 1e6))
                movie.save_frame(new_image_path, t=(track.timestamped_objects[0].time_offset.seconds + track.timestamped_objects[0].time_offset.microseconds / 1e6), withmask=False)
                
                # Each segment includes timestamped faces that include
                # characteristics of the face detected.
                # Grab the first timestamped face
                timestamped_object = track.timestamped_objects[0]
                box = timestamped_object.normalized_bounding_box
                #print("Bounding box:")
                #print("\tleft  : {}".format(box.left))
                #print("\ttop   : {}".format(box.top))
                #print("\tright : {}".format(box.right))
                #print("\tbottom: {}".format(box.bottom))
                
                # Opens a image in RGB mode
                im = Image.open(new_image_path)

                
                # Size of the image in pixels (size of original image)
                width, height = im.size
                
                crop_image_path = os.path.join(output,"image"+str(i)+"cropped.jpg")
                face_path = os.path.join(faces_dir,str(i)+".jpg")
                
                # Cropped image of above dimension
                # (It will not change original image)
                crop_im = im.crop((box.left*width, box.top*height, box.right*width, box.bottom*height))
                
                #TODO keep the same ratios and sizes
                """ def AutoResize(self, frame):
        resizeUtils = ResizeUtils()

        height, width, _ = frame.shape

        if height > 500:
            frame = resizeUtils.rescale_by_height(frame, 500)
            self.AutoResize(frame)
        
        if width > 700:
            frame = resizeUtils.rescale_by_width(frame, 700)
            self.AutoResize(frame)
        
        return frame """
                crop_im.save(crop_image_path)
                crop_im.save(face_path)
                
                #TODO - for testing - get the original thumbnail
                
                thumb_path = output+"\\"+"image"+str(i)+"thumb.jpg"
                (Image.open(io.BytesIO(annotation.thumbnail))).save(thumb_path)

    return os.path.join(output,"Faces")

def landmark(faces_dir):
    #Create an FaceLandmarker object.
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    dict = {}
    
    # for every image, detect landmarks and save to python dict
    for face in os.listdir(faces_dir):
        img_path = os.path.join(faces_dir, face)
        id = os.path.splitext(face)[0]
        image = mp.Image.create_from_file(img_path)
        detection_result = detector.detect(image)
        dict[id] = (img_path,[landmark for landmark in detection_result.face_landmarks])
        
    return dict
        
def cluster(dict):

    # Credits: Arian's pyimagesearch for the clustering code
    # Here we are using the sklearn.DBSCAN functioanlity
    # cluster all the facial embeddings to get clusters 
    # representing distinct people

    landmarks_mat = []
    
    for id in dict:
        flat_landmarks = []
        #print(dict[id])
        #print(dict[id][1])
        #print(dict[id][1][0])
        if len(dict[id][1]) != 0:
            for i in range(478):
                flat_landmarks.append(dict[id][1][0][i].x)
                flat_landmarks.append(dict[id][1][0][i].y)
                flat_landmarks.append(dict[id][1][0][i].z)
            landmarks_mat.append(flat_landmarks)
        else:
            for i in range(478):
                flat_landmarks.append(0)
                flat_landmarks.append(0)
                flat_landmarks.append(0)
            landmarks_mat.append(flat_landmarks)
            print("couldn`t find face in image number ", id)
    #print(landmarks_mat)
    
    # cluster the embeddings
    clt = DBSCAN(eps=0.95, metric="euclidean", min_samples=2)
    clt.fit(landmarks_mat)

    # determine the total number of unique faces found in the dataset
    labelIDs = np.unique(clt.labels_)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    print("unique faces: {}".format(numUniqueFaces))

    return clt.labels_

def split_images(labels,faces_path):
    opened_labels = {}
    faces = os.listdir(faces_path)
    for i in range(len(faces)):
        label = labels[i]
        #print(faces[i], " label: ", label)
        if label not in opened_labels:
            opened_labels[label] = 0
            os.mkdir(os.path.join(faces_path,str(label)))
        opened_labels[label] += 1
        src = os.path.join(faces_path,faces[i])
        dst = os.path.join(faces_path,str(label),faces[i])
        shutil.copy(src, dst) 

#detect_faces("12_years_a_slave","gs://kg-movie/12.Y34r5.4.5l4v3.13.br.sdm0v13sp01nt.c0m.mp4")
#detect_faces("model1","gs://kg-movie/model_video1.mp4")
#detect_faces("model2","gs://kg-movie/model2.mp4")

#python "D:\OneDrive\OneDrive - mail.tau.ac.il\python\face_detection\cluster_pipeline.py" Footage gs://kg-movie/Footage.mp4
#python "D:\OneDrive\OneDrive - mail.tau.ac.il\python\face_detection\cluster_pipeline.py" TwoPauz gs://kg-movie/TwoPauz.mp4
