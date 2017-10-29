#Code for Reastaurant Famous For
#Group19

from skimage import feature
import numpy as np
import cv2
from os import listdir
from sklearn.cross_validation import train_test_split
from os.path import join
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
    # storing the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    
    #Describe function firstly  dividing the each image into regions
    #then making the histogram for each image and returning the list
    #of histograms 
    def describe(self, image, eps=1e-7):
              windowsize_r = 125
              windowsize_c = 100
              # Crop out the window and calculate the histogram
              features=[]
              for r in range(0,image.shape[0], windowsize_r):
                    for c in range(0,image.shape[1], windowsize_c):
                        
                        #Dividing the image into regions
                        window = image[r:r+windowsize_r,c:c+windowsize_c]
                        
                        #local_binary_pattern()
                        #window: region of grayscale image
                        #numPoints: Number of circularly symmetric neighbour set points
                        #radius: Radius of circle
                        #‘nri_uniform’: non rotation-invariant uniform patterns variant
                        
                        lbp = feature.local_binary_pattern(window, self.numPoints,
                                                           self.radius, method="nri_uniform")
                        
                        #histogram():
                        #lbp.ravel(): Return a contiguous flattened array
                        #bins: it defines the number of equal-width bins in the given range
                        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints*(self.numPoints-1)+3),range=(0, self.numPoints*(self.numPoints-1) + 2))
                        
                        #astype(): Copy of the array, cast to a 'float' type.
                        hist = hist.astype("float")
                        hist /= (hist.sum() + eps)
                        features.append(hist)

              return np.array(features).reshape(-1,)
  
#imageDataset: It is a list that storing the path of each image.
#labels: It is a list that storing the label of each image 
#corresponding to the path that stored in image Dataset list.
#foodLabels: It is a list that storing the labels of distinct food category.    
imageDataset = []
labels = []
foodLabels=[]

#loadImages: This function loading the path and corresponding label into
#respective list.
def loadImages(dirPath,label):
    foodLabels.append(label)
    for fileName in listdir(dirPath):
        img=join(dirPath,fileName)
        if img is not None:
            imageDataset.append(img)
            labels.append(label)
    return imageDataset

#This all are the path and label of our image dataset, we are loading the images from
#this provided path.
chocoDir='D:\\Python\\Project\\food_data_set\\chocolate_cake'
loadImages(chocoDir,'chocolate_cake')

sandwichDir='D:\\Python\\Project\\food_data_set\\sandwich'
loadImages(sandwichDir,'sandwich')

donutDir='D:\\Python\\Project\\food_data_set\\donuts'
loadImages(donutDir,'donuts')

frenchDir='D:\\Python\\Project\\food_data_set\\french_fries'
loadImages(frenchDir,'french_fries')

friedDir='D:\\Python\\Project\\food_data_set\\fried_rice'
loadImages(friedDir,'fried_rice')

nachoDir='D:\\Python\\Project\\food_data_set\\nachos'
loadImages(nachoDir,'nachos')

onionDir='D:\\Python\\Project\\food_data_set\\onion_rings'
loadImages(onionDir,'onion_rings')

pizzaDir='D:\\Python\\Project\\food_data_set\\pizza'
loadImages(pizzaDir,'pizza')

samosaDir='D:\\Python\\Project\\food_data_set\\samosa'
loadImages(samosaDir,'samosa')

spaghettiDir='D:\\Python\\Project\\food_data_set\\spaghetti'
loadImages(spaghettiDir,'spaghetti')

steakDir='D:\\Python\\Project\\food_data_set\\steak'
loadImages(steakDir,'steak')

sushiDir='D:\\Python\\Project\\food_data_set\\sushi'
loadImages(sushiDir,'sushi')

#train_test_split: This function splitting the imageDataset into trainingData 
#and testingData with ratio 70:30
(trainingData,testingData,trainingLabels,testingLabels)= train_test_split(imageDataset, labels, test_size=0.30, random_state=4) 

#Initialize the local binary patterns descriptor along with
#the data and label lists
desc = LocalBinaryPatterns(8, 1)
data = []

tot=[]
lab=[]
i=0
# Loop over the training images
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


#AdaBoostClassifier: The model takes a form of histogram representation
# of the images, based on the collection of its local features. 
#then apply AdaBoost on the histogram and model is trained with the training data.
#base_estimator: The base estimator from which the boosted ensemble is built.
#Support for sample weighting is required, as well as proper classes_ and
# n_classes_ attributes.(default=DecisionTreeClassifier)
#n_estimators: he maximum number of estimators at which boosting is terminated.

model = AdaBoostClassifier(n_estimators=100)

#model = AdaBoostClassifier(#base_estimator=model,algorithm='SAMME',n_estimators=100)


#model.fit(): It trains the model for a fixed number of iterations on a dataset.
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
     

#score(): This fuction calculates the accuracy of model.
accuracy=model.score(test_data,testingLabels)

#predict(): This function gives us the predicted labels corresponding to 
#the classified class. 
predictedLabels=model.predict(test_data)

ipred=0
predLabels=[]

#loop for finding number of correctly and incorrectly predicted images.
for i in range (0,len(testingLabels)):
    predLabels.append(predictedLabels[i]);
    if testingLabels[i] != predictedLabels[i]:
        ipred=ipred+1

cpred=len(predictedLabels)-ipred

print("\n")

#print statements for displaying total number of training images
print("Total training images :",len(trainingData),sep="\t")


#print statements for displaying total number of test images
print("Total testing images :",len(testingData),sep="\t")


#print statements for displaying number of correctly predicted test images.
print("Number of images predicted correctly :",cpred,sep="\t")


#print statements for displaying number of incorrectly predicted test images.
print("Number of images predicted incorrectly :",ipred,sep="\t")


#
clf = AdaBoostClassifier(n_estimators=50)
scores = cross_val_score(clf,tot,lab, cv=10)

avg_accuracy = scores.mean()

#print(scores.mean(), scores.std() * 2)


#print statements for displaying the accuracy of classification model.
print("Accuracy of prediction:",accuracy*100,"%")

#Storing the number of images for each food label
labelCounts=[]

#function for storing label counts corresponding the labels that are present
#foodLabels list
def setLabCount(labs):
    labelCounts.append(predLabels.count(labs));

print("\n")
print("The categories of food items we consider in our project are :")
#loop over each element of foodlabels. 
for i, val in enumerate(foodLabels):
    setLabCount(val);
    print(i+1,val,sep="\t");

#maxLabCount: It contains the count of maximum number of images.
maxLabCount=max(labelCounts);

#It is for storing threshold value
threshValue = len(testingData)/len(foodLabels);

#minThreshValue: It is storing the minimum threshold value and 
#we are assuming that is 10% of maximum number of image.
minThreshValue=maxLabCount-0.1*maxLabCount;

#This list contains the list of all the famous food labels that are greater than the
#minimum threshold value. 
selFood=[];

#Function for finding the list of famous food labels using minimum threshold value.
def setFamFood(i,foodCount):
    if foodCount >= minThreshValue:
        selFood.append(foodLabels[i]);
              
#loop over each element of labelCounts.
for i, val in enumerate(labelCounts):
    setFamFood(i,val);

print("\n")
print("\n")
print("The Restaurant is famous for following food items :")

#loop over displaying the list of food labels corresponding to the dataset 
#that are famous for given restaurant 
for i, val in enumerate(selFood):
    print(val,end='  ');

print("\n")
cv2.waitKey(0)
