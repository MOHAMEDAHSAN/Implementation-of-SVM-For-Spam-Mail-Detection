# EXP9 - Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2. Import the python pandas library as pd
3. Read the contents of the Spam csv file
4. Display the first 5 rows of the dataset using head()
5. Assign x as v1 values and y as v2 values
6. From sklearn library select the feature extraction and import CountVectorizer
7. CountVectorizer will convert the Text to Numerical Data
8. From sklearn library import Support Vector Classifier (ie. SVC)
9. Predict the x_test using SVC
10. Print the accuracy of the SVM Model 11.Stop the program

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: S MOHAMED AHSAN
RegisterNumber:  212223240089
*/
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data = pd.read_csv(file,encoding='Windows-1252')

data.head()
data.info()
data.isnull().sum()

x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
#### 1. Result output
![Screenshot 2024-05-04 185944](https://github.com/MOHAMEDAHSAN/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331378/902d919e-dd64-4cc7-b101-5e834755b6f3)
#### 2. data.head()

![Screenshot 2024-05-04 185953](https://github.com/MOHAMEDAHSAN/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331378/185b5cf3-e7b3-4b06-8053-ada4c77eb504)
#### 3. data.info()
![Screenshot 2024-05-04 190000](https://github.com/MOHAMEDAHSAN/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331378/7d7befdb-08ef-4e59-aa53-f01e004926a6)
#### 4. data.isnull().sum()
![Screenshot 2024-05-04 190005](https://github.com/MOHAMEDAHSAN/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331378/9e2c9d80-ff09-41e7-9076-625bc11116a5)
#### 5. Y_prediction value

![Screenshot 2024-05-04 190023](https://github.com/MOHAMEDAHSAN/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331378/4c091d5e-e2bb-469f-926b-68cc04292b5d)
#### 6. Accuracy value

![Screenshot 2024-05-04 190016](https://github.com/MOHAMEDAHSAN/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331378/6cae58ed-cef9-434d-b1a2-74afa36b2ecb)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
