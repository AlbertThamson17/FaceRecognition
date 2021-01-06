import os
import cv2
import math
import numpy as np

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_rootpath(root):
    name = os.listdir(root)
    return name
    
def get_classid(rootpath, trainnames):
    
    face = []
    faceclass = []
    for idx, trainname in enumerate(trainnames):
        namepath = rootpath + '/' + trainname

        for img_path in os.listdir(namepath):
            imgpath = namepath + '/' + img_path
            img = cv2.imread(imgpath)
            face.append(img)
        
            faceclass.append(idx)

    return face, faceclass
        
def detect_train_grayfilter(images, imageclasses):
    face = []
    faceclass = []
    for idx, image in enumerate(images):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected = classifier.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
        if(len(detected) < 1):
            continue
        for face_rect in detected:
            x, y, w, h = face_rect
            faceimg = img_gray[y:y+w, x:x+h]
            face.append(faceimg)
            faceclass.append(idx/10)
    return face, faceclass

def detect_filter_testfaces(images):
        crop = []
        rec = [] 
        for img_bgr in images:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            detected = classifier.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors=5)
            if (len(detected) < 1):
                continue
            for face_rect in detected:
                x, y, h, w = face_rect
                facegray = img_gray[y:y+w, x:x+h]
                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 0, 255), 1)
                rec.append(img_bgr)
                crop.append(facegray)

        return crop, rec

def train_img(train_facegray, imageclasses):
    
    facerecognizer = cv2.face.LBPHFaceRecognizer_create()
    facerecognizer.train_img(train_facegray, np.array(imageclasses))
    
    return facerecognizer

def get_testimages(test_dataroot):
    
    imagetest = []
    for image_path in os.listdir(test_dataroot):
        imgpath = test_dataroot + '/' + image_path
        img_bgr = cv2.imread(imgpath)
        imagetest.append(img_bgr)

    return imagetest
    
def predict(facerecognizer, test_gray):
   
    pred = []
    for img_gray in test_gray:
        detected = classifier.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors=5)
        if (len(detected) < 1):
            continue
        res, loss = facerecognizer.predict(img_gray)
        loss = math.floor(loss * 100) / 100
        pred.append(res)

    return pred

def draw_prediction_results(result, test_image, test_rectface, train_names, resultsize):
    
    img = []
    text = []
    for i in result:
        text.append(train_names[i])
    for idx,image in enumerate(test_rectface):
        cv2.putText(image, text[idx], (resultsize, resultsize), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        img.append(image)
    return img

def view(imagelist, size):
    
    min_height = min(im.shape[0] for im in imagelist)
    image_resize = [cv2.resize(im, (int(im.shape[1] * size / im.shape[0]), min_height), interpolation=cv2.INTER_CUBIC) for im in imagelist]
    img = cv2.hconcat(image_resize)
    cv2.imshow('Image Detected', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    train_rootpath = "C:\Users\Albert\OneDrive\Documents\Semester 5\Computer vision\Project Lab\dataset\train"

    train_names = get_rootpath(train_rootpath)
    train_image_list, image_classes_list = get_classid(train_rootpath, train_names)

    train_grayface, filtered = detect_train_grayfilter(train_image_list, image_classes_list)

    facerecognizer = train_img(train_grayface, filtered)

    test_rootpath = "C:\Users\Albert\OneDrive\Documents\Semester 5\Computer vision\Project Lab\dataset\test"


    test_image = get_testimages(test_rootpath)
    test_grayface, test_rectface = detect_filter_testfaces(test_image)
    
    predicted = predict(facerecognizer, test_grayface)
    predicted_image = draw_prediction_results(predicted, test_image, test_rectface, train_names, 200)
    
    view(predicted_image, 200)