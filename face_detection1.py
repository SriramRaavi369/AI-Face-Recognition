import cv2,numpy,os
size=4
haar_file="haarcascade_frontalface_default.xml"
datasets='datasets'
print("Training")
images,labels,names,id=[],[],{},0
for subdirs,dirs,files in os.walk(datasets):
    for subdir in dirs:
        names[id]=subdir
        subjectpath=os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path=subjectpath+'/'+filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id+=1
(width,height)=(130,100)
images,labels=[numpy.array(lis) for lis in [images,labels]]
model=cv2.face.LBPHFaceRecognizer_create()
#model=cv2.face.FischerFaceRecognizer_create()
model.train(images,labels)
face_cascade=cv2.CascadeClassifier(haar_file)
webcam=cv2.VideoCapture(0)
cnt=0
while True:
    (_,im)=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face=gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(width,height))
        prediction=model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
        if prediction[1]<800:
            #cv2.putText(im,'%s-%.0f'%(names[prediction[0]]))
            print(names[prediction[0]])
            cnt=0
        else:
            cnt+=1
            if(cnt>100):
                print("uknown person")
                cv2.imwrite("input.jpg",im)
                cnt=0
    cv2.imshow("opencv",im)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()
