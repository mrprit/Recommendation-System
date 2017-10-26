from skimage import feature
import numpy as np
import cv2
from os import listdir
from sklearn.model_selection import train_test_split
from os.path import join
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
    # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    
    def describe(self, image, eps=1e-7):
              windowsize_r = 125
              windowsize_c = 100
              # Crop out the window and calculate the histogram
              features=[]
              for r in range(0,image.shape[0], windowsize_r):
                    for c in range(0,image.shape[1], windowsize_c):
                        window = image[r:r+windowsize_r,c:c+windowsize_c]
                        lbp = feature.local_binary_pattern(window, self.numPoints,
                                                           self.radius, method="nri_uniform")
                        
                        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints*(self.numPoints-1)+3),range=(0, self.numPoints*(self.numPoints-1) + 2))
                        hist = hist.astype("float")
                        hist /= (hist.sum() + eps)
                        features.append(hist)

              return np.array(features).reshape(-1,)
      
imageDataset = []
labels = []
def loadImages(dirPath,label):
    for fileName in listdir(dirPath):
        img=join(dirPath,fileName)
        if img is not None:
            imageDataset.append(img)
            labels.append(label)
    return imageDataset

pastaDir='/Users/prajith/Desktop/Pritesh/Dataset/Pasta'
loadImages(pastaDir,'Pasta')

pizzaDir='/Users/prajith/Desktop/Pritesh/Dataset/Pizza'
loadImages(pizzaDir,'Pizza')

burgerDir='/Users/prajith/Desktop/Pritesh/Dataset/Burger'
loadImages(pastaDir,'Burger')

noodlesDir='/Users/prajith/Desktop/Pritesh/Dataset/Noodles'
loadImages(pizzaDir,'Noodles')

(trainingData,testingData,trainingLabels,testingLabels)= train_test_split(imageDataset, labels, test_size=0.40, random_state=4) 
# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(8, 1)
data = []

tot=[]
lab=[]
i=0
# loop over the training images
for imagePath in trainingData:
# load the image, convert it to grayscale, and describe it
     image = cv2.imread(imagePath)
     image = cv2.resize(image,(500,500))
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     hist = desc.describe(gray)
     # extract the label from the image path, then update the
     # label and data lists
     data.append(hist)
     tot.append(hist)
     lab.append(trainingLabels[i])
     i=i+1

# train a Linear SVM on the data
#model = svm.SVC(C=100.0,decision_function_shape='ovr', kernel='rbf')
model = AdaBoostClassifier(#base_estimator=model,algorithm='SAMME',
                           n_estimators=100)
model.fit(data, trainingLabels)

test_data=[]
i=0
# loop over the testing images
for imagePath in testingData:
# load the image, convert it to grayscale, describe it,
# and classify it
     image = cv2.imread(imagePath)
     image = cv2.resize(image,(500,500))
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     hist = desc.describe(gray)
     test_data.append(hist)
     tot.append(hist)
     lab.append(testingLabels[i])
     i=i+1
     
accuracy=model.score(test_data,testingLabels)
predictedLabels=model.predict(test_data)
ipred=0 
print(" ","S.No.",end="")
print("   ","Actual Label","Predicted Label",sep="\t")
for i in range (0,len(testingLabels)):
    print("  ",i,end="")
    print("    ",testingLabels[i],predictedLabels[i],sep="\t\t")
    if testingLabels[i] != predictedLabels[i]:
        image = cv2.imread(testingData[i])
        cv2.namedWindow('image %d' % (i,),cv2.WINDOW_AUTOSIZE)
        cv2.imshow('image %d' % (i,),image)
        ipred=ipred+1

cpred=len(predictedLabels)-ipred

print("\n")
print("Total training images :",len(trainingData),sep="\t")
print("Total testing images :",len(testingData),sep="\t")
print("Number of images predicted correctly :",cpred,sep="\t")
print("Number of images predicted incorrectly :",ipred,sep="\t")
print("Accuracy of prediction:",accuracy*100,"%")

#model = svm.SVC(C=100.0,kernel='linear')
clf = AdaBoostClassifier(#base_estimator=model,algorithm='SAMME',
                         n_estimators=50)
scores = cross_val_score(clf,tot,lab, cv=10)
print(scores.mean(), scores.std() * 2)

cv2.waitKey(0)
