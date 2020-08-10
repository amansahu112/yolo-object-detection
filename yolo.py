import cv2
import numpy as np

classesFiles='coco.names'
classNames=[]
with open(classesFiles,'r') as f:
    classNames=f.read().rstrip('\n').split('\n')

# print(classNames)

modelconfig='darknet_yolov3-tiny.cfg'
modelweight='yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelconfig, modelweight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

cap = cv2.VideoCapture(0)
wt=320
thresold_confidence= 0.5
nmsThreshold=0.3

def findobjects(outputs,img):
    hT,wT,cT=img.shape
    bbox=[]
    classIds=[]
    confs=[]

    for outpput in outputs:
        for det in outpput:
            scores=det[5:]
            classId=np.argmax(scores)
            confidence=scores[classId]
            if confidence > thresold_confidence:
                w,h=int(det[2]*wT),int(det[3]*hT)
                x,y=int((det[0]*wT)-w/2),int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print(len(bbox))
    indices=cv2.dnn.NMSBoxes(bbox,confs,thresold_confidence,nms_threshold=nmsThreshold)
    for i in indices:
        i=i[0]
        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,255),2)

while True:
    success, img = cap.read()
    blob= cv2.dnn.blobFromImage(img,1/255.0,(wt,wt),[0,0,0],1,crop=False)
    net.setInput(blob)
    layerNames= net.getLayerNames()
    # print(net.getUnconnectedOutLayers())
    outputNames=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    outputs=net.forward(outputNames)
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    findobjects(outputs,img)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()