import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

def yolo_filter_boxes(box_confidence,boxes,box_class_probs,threshold=.6):
    #计算每个class的概率
    box_scores=box_confidence*box_class_probs
    print('box_scores:',box_scores)
    box_classes=K.argmax(box_scores,axis=-1)#获得所有box得分最高的类别
    box_class_scores=K.max(box_scores,axis=-1,keepdims=False)#获得每个box最高的得分
    print('box_class_scores:',box_class_scores)
    filtering_mask=box_class_scores>=threshold
    print('filtering_mask',filtering_mask)
    scores=tf.boolean_mask(box_class_scores,filtering_mask)
    boxes=tf.boolean_mask(boxes,filtering_mask)

    print('boxes:',boxes)
    classes=tf.boolean_mask(box_classes,filtering_mask)
    return scores,boxes,classes

def iou(box1,box2):
    xi1=max(box1[0],box2[0])
    yi1=max(box1[1],box2[1])
    xi2=min(box1[2],box2[2])
    yi2=min(box1[3],box2[3])
    inter_area=(yi2-yi1)*(xi2-xi1)

    box1_area=(box1[3]-box1[1])*(box1[2]-box1[0])
    box2_area=(box2[3]-box2[1])*(box2[2]-box2[0])
    union_area=box1_area+box2_area-inter_area

    iou=inter_area/union_area

    return iou

def yolo_non_max_suppression(scores,boxes,classes,max_boxes=10,iou_threshold=0.5):
    max_boxes_tensor=K.variable(max_boxes,dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

    nms_indices=tf.image.non_max_suppression(boxes=boxes,scores=scores,max_output_size=max_boxes_tensor,iou_threshold=iou_threshold)
    print('nms_indices:',nms_indices)

    scores=K.gather(scores,nms_indices)
    boxes=K.gather(boxes,nms_indices)
    classes=K.gather(classes,nms_indices)

    return scores,boxes,classes

def yolo_eval(yolo_outputs,images_shape=(720.,1280.),max_boxes=10,score_threshold=.6,iou_threshold=0.5):
    box_confidence,box_xy,box_wh,box_class_probs=yolo_outputs

    boxes=yolo_boxes_to_corners(box_xy,box_wh)
    print('boxes:',boxes)

    scores,boxes,classes=yolo_filter_boxes(box_confidence,boxes,box_class_probs,score_threshold)
    boxes=scale_boxes(boxes,images_shape)

    scores,boxes,classes=yolo_non_max_suppression(scores,boxes,classes,max_boxes,iou_threshold)

    return  scores,boxes,classes

sess=K.get_session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)
yolo_model=load_model("model_data/yolo.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)


def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    ### START CODE HERE ### (≈ 1 line)
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    ### END CODE HERE ###

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)

    return out_scores, out_boxes, out_classes
out_scores, out_boxes, out_classes = predict(sess, "test.jpg")

