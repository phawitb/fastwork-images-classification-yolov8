import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import json 

def list_images_in_folder(folder_path):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                image_files.append(os.path.join(root, file))
    return image_files

def pred(img_path):
    im = Image.open(img_path)
    rs = model.predict(source=[im])
    result = {}
    for r in rs:
        for i,t in enumerate(r.probs.top5):
            result[i] = {
                'name' : r.names[t],
                'confidence' : float(r.probs.top5conf[i])
            }
    return result[0]['name'] 

# config ------------------------

model = YOLO('runs/classify/train4/weights/best.pt')
class_names = ['healthy', 'narrow_brown_spot_disease','rice_blast_disease)' ]
data_floders = ['train','val','test']

#---------------------------------------------

directory = 'evaluations'
if not os.path.exists(directory):
    os.makedirs(directory)

results = {}
for data_floder in data_floders:
    # data_floder = 'train'
    folder_path = f'data/{data_floder}'
    images = list_images_in_folder(folder_path)

    #create y_true,imgs_path
    y_true = []
    imgs_path = []
    for img in images:
        imgs_path.append(img)
        cls = img.split('/')[2]
        cls_index = class_names.index(cls)
        y_true.append(cls_index)
        
    #find y_pred
    y_pred = []
    for img_path in imgs_path:
        cls = pred(img_path)
        cls_index = class_names.index(cls)
        y_pred.append(cls_index)

    #find confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=45)
    plt.title(f'Confusion Matrix with {data_floder} set')
    plt.savefig(f'{directory}/confusion_matrix_{data_floder}.png', bbox_inches='tight')
    # plt.show()

    #find precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    results[data_floder] = {
        'precision' : precision,
        'recall' : recall,
        'f1_score' : f1_score
    }

    #find ROC
    n_classes = 3
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    colors = ['blue', 'orange', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {data_floder} set')
    plt.legend(loc="lower right")
    plt.savefig(f'{directory}/roc_{data_floder}.png')
    # plt.show()

    

# Writing the data dictionary to a JSON file
with open(f'{directory}/evauation.json', 'w') as file:
    json.dump(results, file, indent=4)

