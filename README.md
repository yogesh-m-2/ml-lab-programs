# ml-lab-programs

<h3>prog1<h3>
  import csv
def read_data(filename):
    with open(filename,'r') as csvfile:
        datareader=csv.reader(csvfile,delimiter=',')
        headers=next(datareader)
        traindata=[]
        for row in datareader:
            traindata.append(row)
        return(traindata,headers)                   

def finds():
    dataset,features=read_data('data1.csv') 
    rows=len(dataset);
    cols=len(dataset[0]);
    flag=0
    for x in range(0,rows):
        t=dataset[x]
        if t[-1]=='1'and flag==0:
            flag=1
            h=dataset[x]
        elif t[-1]=='1':
            for y in range(cols):
                if h[y]!=t[y]:
                    h[y]='?'
    print_hypo(h)
    
def print_hypo(h):
    print('<',end='')
    for i in range(0,len(h)-1):
        print(h[i],end=',')
    print('>') 
finds()   
  <h3>prog4<h3>
    import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float) 
X = X/np.amax(X, axis=0) 
class Neural_Network(object): 
  def __init__ (self):
        self.inputSize=2
        self.outputSize = 1
        self.hiddenSize = 3

        
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

  def forward(self, X):

    self.z = np.dot(X, self.W1) 
    self.z2 = self.sigmoid(self.z) 
    self.z3 = np.dot(self.z2, self.W2) 
    o = self.sigmoid(self.z3)
    return o

  def sigmoid(self,  s): 
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    return s * (1 - s)

  def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error*self.sigmoidPrime(o)

        self.z2_error = self.o_delta.dot(self.W2.T) 
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) 

        self.W1 += X.T.dot(self.z2_delta) 
        self.W2 += self.z2.T.dot(self.o_delta)

  def train (self, X, y): 
    o = self.forward(X) 
    self.backward(X, y, o)

NN = Neural_Network()
for i in range(10): # trains the NN 1,000 times 
    print(i)
    print ("Input: \n" + str(X))
    print ("Actual Output: \n" + str(y))
    print ("Predicted Output: \n" + str(NN.forward(X)))
    print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))) )# mean sum squared loss
    print ("\n")
    NN.train(X, y)
    <h3>prog5<h3>
  import csv
import random
import math

def loadCsv(filename):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset
 
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]
 
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated
 
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)
 
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries
 
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries
 
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
 
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
 
def main():
	filename = 'naivedata.csv'
	splitRatio = 0.67
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy of the classifier is : {0}%'.format(accuracy))
 
main()
      <h3>prog6<h3>
  import pandas as pd
msg=pd.read_csv('naivetext1.csv',header=None,names=['message','label'],)
print('The dimensions of the dataset',msg.shape)
print('------------------------------------------------------------------------')
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
y=msg.labelnum

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=1)
print('Dimensions of train and test sets')
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)
#output of count vectorizer is a sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer() 
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm=count_vect.transform(xtest)
#training naive bayes(NB) classifier on training data
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm,ytrain)
predicted = clf.predict(xtest_dtm)
print('\nClassification results of testing samples are given below')
print('------------------------------------------------------------------------')
for doc,p in zip(xtest,predicted):
  pred='pos' if p==1 else 'neg'
  print('%s->%s'%(doc,pred))
print('------------------------------------------------------------------------')
#printing accuracy metrics
from sklearn import metrics
print('Accuracy metrics')
print('Accuracy of the classifer is',metrics.accuracy_score(ytest,predicted))
print('------------------------------------------------------------------------')
print('Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('------------------------------------------------------------------------')
print('Recall and Precison ')
print(metrics.recall_score(ytest,predicted))
print(metrics.precision_score(ytest,predicted))
        <h3>prog7<h3>
   import numpy as np
 import pandas as pd
 import csv
 from pgmpy.estimators import MaximumLikelihoodEstimator
 from pgmpy.models import BayesianModel
 from pgmpy.inference import VariableElimination

 #Read the attributes
 lines = list(csv.reader(open('data7_names.csv', 'r')));
 attributes = lines[0]

 heartDisease = pd.read_csv('data7_heart.csv', names = attributes)
 heartDisease = heartDisease.replace('?', np.nan)
 # Model Baysian Network
 model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('sex', 'trestbps'),
 ('exang', 'trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),
 ('heartdisease','restecg'),('heartdisease','thalach'),('heartdisease','chol')])

 # Learning CPDs using Maximum Likelihood Estimators
 print('\n Learning CPDs using Maximum Likelihood Estimators...');
 model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

 # Inferencing with Bayesian Network
 print('\n Inferencing with Bayesian Network:')
 HeartDisease_infer = VariableElimination(model)

 # Computing the probability of bronc given smoke.
 print('\n 1.Probability of HeartDisease given Age=28')
 q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 28})
 print(q)

 print('\n2. Probability of HeartDisease given chol (Cholestoral) =100')
 q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': 100})
 print(q)
          <h3>prog8<h3>
  import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn import preprocessing

iris=datasets.load_iris()
x=pd.DataFrame(iris.data)
x.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y=pd.DataFrame(iris.target)
y.columns=['Targets']

model=KMeans(n_clusters=3)
model.fit(x)

plt.figure(figsize=(14,14))
colormap=np.array(['red','lime','black'])
plt.subplot(2,2,1)
plt.scatter(x.Petal_Length,x.Petal_Width,c=colormap[y.Targets],s=40)
plt.title('Real Clusters')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.subplot(2,2,2)
plt.scatter(x.Petal_Length,x.Petal_Width,c=colormap[model.labels_],s=40)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('K-Means Clustering')

scaler=preprocessing.StandardScaler()
scaler.fit(x)
xsa=scaler.transform(x)
xs=pd.DataFrame(xsa,columns=x.columns)
from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(n_components=3)
gmm.fit(xs)
gmm_y=gmm.predict(xs)

plt.subplot(2,2,3)
plt.scatter(x.Petal_Length,x.Petal_Width,c=colormap[gmm_y],s=40)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('GMM clustering')
print('Observation: The GMM using EM algorithm based clustering matched the true tables more closely than the KMeans')
            <h3>prog9<h3>
  from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import datasets
iris=datasets.load_iris()
iris_data=iris.data
iris_lables=iris.target
x_train,x_test,y_train,y_test=train_test_split(iris_data,iris_lables,test_size=0.30)
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print('Confusion Matrix is as follows:')
print(confusion_matrix(y_test,y_pred))
print('Accuracy Matrix is as follows:')
print(classification_report(y_test,y_pred))

              <h3>prog10<h3>
              from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy.linalg
from scipy.stats.stats import pearsonr

def kernel(point,xmat,k):
  m,n=shape(xmat)
  weights=mat(eye((m)))
  for j in range(m):
    diff=point-x[j]
    weights[j,j]=exp(diff*diff.T/(-2.0*k*2))
  return weights

def localWeight(point,xmat,ymat,k):
  wei=kernel(point,xmat,k)
  w=(x.T*(wei*x)).I*(x.T*(wei*ymat.T))
  return w

def localWeightedRegression(xmat,ymat,k):
  m,n=shape(xmat)
  ypred=zeros(m)
  for i in range(m):
    ypred[i]=xmat[i]*localWeight(xmat[i],xmat,ymat,k)
  return ypred

data=pd.read_csv('tips.csv')
bill=array(data.total_bill)
tip=array(data.tip)

mbill=mat(bill)
mtip=mat(tip)
m=shape(mbill)[1]
one=mat(ones(m))
x=hstack((one.T,mbill.T))

ypred=localWeightedRegression(x,mtip,0.5)
SortIndex=x[:,1].argsort(0)
xsort=x[SortIndex][:,0]

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(bill,tip,color='green')
ax.plot(xsort[:,1],ypred[SortIndex],color='red',linewidth=5)
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show();
              
