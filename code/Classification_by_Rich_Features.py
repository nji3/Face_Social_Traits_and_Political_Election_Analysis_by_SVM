import numpy as np 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
from scipy.optimize import minimize
import glob
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedShuffleSplit, ShuffleSplit
from skimage import exposure
from skimage import feature
import matplotlib.pyplot as plt

### Part 1
## Task 1.1 Classification by Landmarks
# Read Landmarks of all 491 images and Trait Annotations
all = '/Users/nanji/Desktop/UCLA/2018Fall/stat231/project3_code_and_data/train-anno.mat'
all_dic = scipy.io.loadmat(all)
all_landmarks = all_dic['face_landmark'] # 491x160
all_traits_value = all_dic['trait_annotation'] # 491x14

# Use Min-Max scale to scale the landmarks to 0-1 range
for i in range(491):
    row_min = min(all_landmarks[i,:])
    row_max = max(all_landmarks[i,:])
    all_landmarks[i,:] = (all_landmarks[i,:]-row_min)/(row_max-row_min)
# Convert the numerical traits to be binary classes
for i in range(491):
    row_mean = np.mean(all_traits_value[i,:])
    all_traits_value[i,:] = all_traits_value[i,:] - row_mean

all_traits = np.sign(all_traits_value)

# Perform SVC on the scaled landmarks for each traits and do k-fold cross-validations
# Tune the penalty parameter C and the kernel coefficient parameter gamma
def svm_kfold_crossv(X_train,X_test,y_train,y_test,k=5,c_max=5,gamma_max=0.1,n_block=5):
    C = np.linspace(1,c_max,n_block)
    Gamma = np.linspace(0.01,gamma_max,n_block)
    clf = GridSearchCV(estimator=SVC(C=1), param_grid=dict(C=C,gamma=Gamma),cv=k,n_jobs=-1)
    clf.fit(X_train, y_train)
    slc_pair = [clf.best_estimator_.C,clf.best_estimator_.gamma]
    clf_slect = SVC(C=slc_pair[0],gamma=slc_pair[1])
    clf_slect.fit(X_train,y_train)
    return slc_pair, clf_slect.score(X_train, y_train), sum(clf_slect.predict(X_test)==y_test)/len(y_test)

def svr_kfold_crossv(X_train,X_test,y_train,y_test,k=5,c_max=5,gamma_max=0.1,eps_max=0.2,n_block=5):
    C = np.linspace(1,c_max,n_block)
    Gamma = np.linspace(0.01,gamma_max,n_block)
    epsilons = np.linspace(0.1,0.2,2)
    clf = GridSearchCV(estimator=SVR(C=1), param_grid=dict(C=C,gamma=Gamma,epsilon=epsilons),\
    cv=k,n_jobs=-1)
    clf.fit(X_train, y_train)
    slc_pair = [clf.best_estimator_.C,clf.best_estimator_.gamma,clf.best_estimator_.epsilon]
    clf_slect = SVR(C=slc_pair[0],gamma=slc_pair[1],epsilon=slc_pair[2])
    clf_slect.fit(X_train,y_train)
    train_mse = sum(np.square(clf_slect.predict(X_train)-y_train))/X_train.shape[0]
    test_mse = sum(np.square(clf_slect.predict(X_test)-y_test))/X_test.shape[0]
    return slc_pair, train_mse, test_mse

## Task 1.2 Classification by Landmarks
# Read images
images = [cv2.imread(file) for file in\
glob.glob('/Users/nanji/Desktop/UCLA/2018Fall/stat231/project3_code_and_data/img/*.jpg')]

# Calculate the Hog features for all 491 images and Combine the hogs features with landmarks
Hogs,hogImage= feature.hog(images[0], orientations=8, pixels_per_cell=(32, 32),cells_per_block=(2, 2),\
transform_sqrt=True, block_norm="L1",visualise=True)
for im in images:
    h,_=feature.hog(im, orientations=8, pixels_per_cell=(32, 32),cells_per_block=(2, 2),\
    transform_sqrt=True, block_norm="L1",visualise=True)
    Hogs = np.row_stack((Hogs, h))
Hogs = Hogs[1:492,:]
hog_dat = np.column_stack((all_landmarks,Hogs)) # 491x6432

X_train, X_test, y_train, y_test = train_test_split(hog_dat, all_traits, test_size=0.3, random_state=231)
svc_train_accuracy = []
svc_test_accuracy = []
svc_params = []

for t in range(14):
    param,train_accuracy,test_accuracy = svm_kfold_crossv(X_train,X_test,y_train[:,t],y_test[:,t])
    svc_train_accuracy.append(train_accuracy)
    svc_test_accuracy.append(test_accuracy)
    svc_params.append(param)
    print('%d is done' %t)

svm_res = [svc_params,svc_train_accuracy,svc_test_accuracy]
np.save('1.2SVM_Res.npy',svm_res)
svm_res = np.load('1.2SVM_Res.npy')
svc_params = svm_res[0]
svc_train_accuracy = svm_res[1]
svc_test_accuracy = svm_res[2]

train_precisions = []
test_precisions = []

for i in range(14):
    clf = SVC(C=svc_params[i][0],gamma=svc_params[i][1])
    clf.fit(X_train,y_train[:,i])
    svc_train_accuracy.append(clf.score(X_train,y_train[:,i]))
    predict = sum(clf.predict(X_test)==y_test[:,i])/len(y_test[:,i])
    svc_test_accuracy.append(predict)
    TP_train = sum(np.bitwise_and(clf.predict(X_train)==y_train[:,i],clf.predict(X_train)==1))
    FP_train = sum(np.bitwise_and(clf.predict(X_train)!=y_train[:,i],clf.predict(X_train)==1))
    train_precisions.append(TP_train/(TP_train+FP_train))
    TP_test = sum(np.bitwise_and(clf.predict(X_test)==y_test[:,i],clf.predict(X_test)==1))
    FP_test = sum(np.bitwise_and(clf.predict(X_test)!=y_test[:,i],clf.predict(X_test)==1))
    test_precisions.append(TP_test/(TP_test+FP_test))

for i in range(14):
    if test_precisions[i]<0.5: test_precisions[i] = 1-test_precisions[i]

# SVR
X_train, X_test, y_train, y_test = train_test_split(hog_dat, all_traits_value, test_size=0.3, random_state=231)
svr_train_mse = []
svr_test_mse = []
svr_params = []

for t in range(14):
    param,train_mse,test_mse = svr_kfold_crossv(X_train,X_test,y_train[:,t],y_test[:,t])
    svr_train_mse.append(train_mse)
    svr_test_mse.append(test_mse)
    svr_params.append(param)
    print('%d is done' %t)

svr_res = [svr_params,svr_train_mse,svr_test_mse]
np.save('1.2SVR_Res.npy',svr_res)
svr_res = np.load('1.2SVR_Res.npy')
svr_params = svr_res[0]
svr_train_mse = svr_res[1]
svr_test_mse = svr_res[2]

# Plot Accuracy
plt.plot(svc_train_accuracy, label="train")
plt.plot(svc_test_accuracy, label="test")
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.xlabel('Traits')
plt.ylabel('Accuracy')
plt.title('1.2 SVC Accuracy')
plt.show()

# Plot Precision
plt.plot(train_precisions, label="train")
plt.plot(test_precisions, label="test")
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.xlabel('Traits')
plt.ylabel('Precision')
plt.title('1.2 SVC Precision')
plt.show()

# Plot MSE
plt.plot(svr_train_mse, label="train")
plt.plot(svr_test_mse, label="test")
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.xlabel('Traits')
plt.ylabel('MSE')
plt.title('1.2 SVR Mean Squared Error')
plt.show()