import numpy as np 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
from scipy.optimize import minimize
import glob
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedShuffleSplit
from skimage import exposure
from skimage import feature

### Part 1
## Task 1.1 Classification by Landmarks
# Read Landmarks of all 491 images and Trait Annotations
all = '*/data/train-anno.mat'
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

# Split the test and training data
# Do the Cross Validation SVM
X_train, X_test, y_train, y_test = train_test_split(all_landmarks, all_traits, test_size=0.3)
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
np.save('*/data/1.1SVM_Res.npy',svm_res)
svm_res = np.load('*/data/1.1SVM_Res.npy')
svc_train_accuracy = svm_res[0]
svc_test_accuracy = svm_res[1]
svc_params = svm_res[2]

# Do the Cross Validation SVR
X_train, X_test, y_train, y_test = train_test_split(all_landmarks, all_traits_value, test_size=0.3)
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
np.save('*/data/1.1SVR_Res.npy',svr_res)
svr_res = np.load('*/data/1.1SVR_Res.npy')
svr_train_mse = svr_res[0]
svr_test_mse = svr_res[1]
svr_params = svr_res[2]

## Task 1.2 Classification by Landmarks
img_path = '*/data/img/M0001.jpg'
im = cv2.imread(img_path)

# Try an example to plot the hog feature
H, hogImage = feature.hog(im, orientations=8, pixels_per_cell=(32,32),cells_per_block=(1, 1),\
transform_sqrt=True, block_norm="L1",visualise=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
cv2.imshow("HOG Image", hogImage)

# Read images
images = [cv2.imread(file) for file in\
glob.glob('*/data/img/*.jpg')]

# Calculate the Hog features for all 491 images and Combine the hogs features with landmarks
Hogs,hogImage= feature.hog(images[0], orientations=8, pixels_per_cell=(32, 32),cells_per_block=(2, 2),\
transform_sqrt=True, block_norm="L1",visualise=True)
for im in images:
    h,_=feature.hog(im, orientations=8, pixels_per_cell=(32, 32),cells_per_block=(2, 2),\
    transform_sqrt=True, block_norm="L1",visualise=True)
    Hogs = np.row_stack((Hogs, h))
Hogs = Hogs[1:492,:]
Hogs = min_max(Hogs)
hog_dat = np.column_stack((all_landmarks,Hogs)) # 491x6432

# Split the test and training data
# Do the Cross Validation
X_train, X_test, y_train, y_test = train_test_split(hog_dat, all_traits, test_size=0.3, random_state=231)
svc_train_accuracy = []
svc_test_accuracy = []
svc_params = []

t = 1
C = np.linspace(0.5,1.5,5)
Gamma = np.linspace(0.15,0.3,5)
clf = GridSearchCV(estimator=SVC(C=1), param_grid=dict(C=C,gamma=Gamma),cv=5,n_jobs=-1)
clf.fit(X_train, y_train[:,t])
clf.best_score_
cl = SVC(C=clf.best_estimator_.C,gamma=clf.best_estimator_.gamma)
cl.fit(X_train, y_train[:,t])
cl.score(X_train, y_train[:,t])
sum(cl.predict(X_test)==y_test[:,t])/len(y_test[:,t])

svc_train_accuracy.append(cl.score(X_train, y_train[:,t]))
svc_test_accuracy.append(sum(cl.predict(X_test)==y_test[:,t])/len(y_test[:,t]))
svc_params.append([clf.best_estimator_.C,clf.best_estimator_.gamma])

for t in range(14):
    param,train_accuracy,test_accuracy = svm_kfold_crossv(X_train,X_test,y_train[:,t],y_test[:,t])
    svc_train_accuracy.append(train_accuracy)
    svc_test_accuracy.append(test_accuracy)
    svc_params.append(param)

svm_res = [svc_params,svc_train_accuracy,svc_test_accuracy]
np.save('*/data/1.2SVM_Res.npy',svm_res)
svm_res = np.load('*/data/1.2SVM_Res.npy')
svc_train_accuracy = svm_res[0]
svc_test_accuracy = svm_res[1]
svc_params = svm_res[2]

# SVR
X_train, X_test, y_train, y_test = train_test_split(hog_dat, all_traits_value, test_size=0.3, random_state=231)
svr_train_mse = []
svr_test_mse = []
svr_params = []

t = 4
C = np.linspace(1,5,5)
Gamma = np.linspace(0.001,0.01,5)
epsilons = np.linspace(0.1,0.2,2)
clf = GridSearchCV(estimator=SVR(C=1), param_grid=dict(C=C,gamma=Gamma,epsilon=epsilons),\
cv=5,n_jobs=-1)
clf.fit(X_train, y_train[:,t])
slc_pair = [clf.best_estimator_.C,clf.best_estimator_.gamma,clf.best_estimator_.epsilon]
clf_slect = SVR(C=slc_pair[0],gamma=slc_pair[1],epsilon=slc_pair[2])
clf_slect.fit(X_train,y_train[:,t])
sum(np.square(clf_slect.predict(X_train)-y_train[:,t]))/X_train.shape[0]
sum(np.square(clf_slect.predict(X_test)-y_test[:,t]))/X_test.shape[0]
sum(np.sign(clf_slect.predict(X_train))==np.sign(y_train[:,t]))/X_train.shape[0]
sum(np.sign(clf_slect.predict(X_test))==np.sign(y_test[:,t]))/X_test.shape[0]

for t in range(14):
    param,train_mse,test_mse = svr_kfold_crossv(X_train,X_test,y_train[:,t],y_test[:,t])
    svr_train_mse.append(train_mse)
    svr_test_mse.append(test_mse)
    svr_params.append(param)
    print('%d is done' %t)

svr_res = [svr_params,svr_train_mse,svr_test_mse]
np.save('*/data/1.2SVR_Res.npy',svr_res)
svr_res = np.load('*/data/1.2SVR_Res.npy')
svr_params = svr_res[0]
svr_train_mse = svr_res[1]
svr_test_mse = svr_res[2]

## Task 2.1 Direct Prediction by Rich Features
# Min-Max Normalization
def min_max(dat):
    for i in range(dat.shape[0]):
        row_min = min(dat[i,:])
        row_max = max(dat[i,:])
        dat[i,:] = (dat[i,:]-row_min)/(row_max-row_min)
    return(dat)

# Function to extract Hog Features
def extract_hog_features(path):
    images = [cv2.imread(file) for file in glob.glob(path)]
    length = len(images)
    Hogs,_= feature.hog(images[0], orientations=8, pixels_per_cell=(32, 32),cells_per_block=(2, 2),\
    transform_sqrt=True, block_norm="L1",visualise=True)
    for im in images:
        h,_=feature.hog(im, orientations=8, pixels_per_cell=(32, 32),cells_per_block=(2, 2),\
        transform_sqrt=True, block_norm="L1",visualise=True)
        Hogs = np.row_stack((Hogs, h))
    Hogs = Hogs[1:(length+1),:]
    return(Hogs)

# Read the landmarks for governors and senators seperately
gov = '*/data/stat-gov.mat'
sen = '*/data/stat-sen.mat'
gov_landmarks = scipy.io.loadmat(gov)['face_landmark']
gov_vote_diff = scipy.io.loadmat(gov)['vote_diff']
sen_landmarks = scipy.io.loadmat(sen)['face_landmark']
sen_vote_diff = scipy.io.loadmat(sen)['vote_diff']
gov_landmarks = min_max(gov_landmarks)
sen_landmarks = min_max(sen_landmarks)

# Extract hog features of governors and senators seperately
gov_im = '*/data/*.jpg'
sen_im = '*/data/*.jpg'
gov_hog = extract_hog_features(gov_im)
sen_hog = extract_hog_features(sen_im)
gov_hog = min_max(gov_hog)
sen_hog = min_max(sen_hog)

# Combine landmarks and hog features
gov_dat = np.column_stack((gov_landmarks,gov_hog)) # 112x6432
sen_dat = np.column_stack((sen_landmarks,sen_hog)) # 116x6432

# According to the election outcome, build new training dataset for the rank SVM
def dat_rankSVM(X,y):
    y = np.sign(y)
    win = X[(y>0)[:,0],:]
    lose = X[(y<0)[:,0],:]
    X1 = win-lose
    X2 = lose - win
    X_new = np.vstack((X1,X2))
    y_new = np.concatenate((np.ones(X1.shape[0]),-1*np.ones(X2.shape[0])))
    return X_new,y_new

# Do K-fold cross validation to find the Parameters we want
gov_new_dat,gov_label = dat_rankSVM(gov_dat,gov_vote_diff)
C = np.linspace(1,5,5)
Gamma = np.linspace(0.01,0.1,5)
clf = GridSearchCV(estimator=SVC(C=1), param_grid=dict(C=C,gamma=Gamma),cv=5,n_jobs=-1)
clf.fit(gov_new_dat, gov_label)
clf.best_score_ # 0.6428571428571429
cl = SVC(C=clf.best_estimator_.C,gamma=clf.best_estimator_.gamma)
scores = cross_val_score(cl, gov_new_dat,gov_label, cv=5)
scores.mean() # 0.6439393939393939

# Do K-fold cross validation to find the Parameters we want
sen_new,sen_label = dat_rankSVM(sen_dat,sen_vote_diff)
C = np.linspace(1,5,5)
Gamma = np.linspace(0.01,0.1,5)
clf = GridSearchCV(estimator=SVC(C=1), param_grid=dict(C=C,gamma=Gamma),cv=5,n_jobs=-1)
clf.fit(sen_new,sen_label)
clf.best_score_ # 0.631578947368421
cl = SVC(C=clf.best_estimator_.C,gamma=clf.best_estimator_.gamma)
scores = cross_val_score(cl, sen_new, sen_label, cv=5)
scores.mean() # 0.6272727272727272

## Task 2.2
# Two layers of SVM
# First layer contains 14 models for 14 traits
# Read the selected 14 models
svr_res = np.load('*/data/1.2SVR_Res.npy')
svr_params = svr_res[0]

gov_dat
sen_dat

# For governors
pred_trait = []
for i in range(14):
    clf = SVR(C=svr_params[i][0],gamma=svr_params[i][1],epsilon=svr_params[i][2])
    clf.fit(hog_dat,all_traits_value[:,i])
    pred_trait.append(clf.predict(gov_dat))
gov_traits = np.array(pred_trait).T
gov_pair_traits,gov_pair_label = dat_rankSVM(gov_traits,gov_vote_diff)
gov_pair_train,gov_pair_test,gov_label_train,gov_label_test=train_test_split(gov_pair_traits,\
gov_pair_label, test_size=0.3, random_state=231)
C = np.linspace(18,18.5,10)
clf = GridSearchCV(estimator=LinearSVC(C=1,fit_intercept=False),\
param_grid=dict(C=C),cv=5,n_jobs=-1)
clf.fit(gov_pair_train,gov_label_train)
clf.best_score_ # 0.5256410256410257
clf.best_estimator_.C # 18.0
cl = LinearSVC(C=clf.best_estimator_.C,fit_intercept=False)
cl.fit(gov_pair_train,gov_label_train)
cl.score(gov_pair_train,gov_label_train) # 0.6410256410256411
sum(cl.predict(gov_pair_test)==gov_label_test)/len(gov_label_test) # 0.5882352941176471
cl.coef_

# For senators
pred_trait = []
for i in range(14):
    clf = SVR(C=svr_params[i][0],gamma=svr_params[i][1],epsilon=svr_params[i][2])
    clf.fit(hog_dat,all_traits_value[:,i])
    pred_trait.append(clf.predict(sen_dat))
sen_traits = np.array(pred_trait).T
sen_pair_traits,sen_pair_label = dat_rankSVM(sen_traits,sen_vote_diff)
sen_pair_train,sen_pair_test,sen_label_train,sen_label_test=train_test_split(sen_pair_traits,\
sen_pair_label, test_size=0.3, random_state=231)
C = np.linspace(4,8,5)
clf = GridSearchCV(estimator=LinearSVC(C=1,fit_intercept=False),\
param_grid=dict(C=C),cv=5,n_jobs=-1)
clf.fit(sen_pair_train,sen_label_train)
clf.best_score_ # 0.5822784810126582
clf.best_estimator_.C # 6.0
cl = LinearSVC(C=clf.best_estimator_.C,fit_intercept=False)
cl.fit(sen_pair_train,sen_label_train)
cl.score(sen_pair_train,sen_label_train) # 0.6582278481012658
sum(cl.predict(sen_pair_test)==sen_label_test)/len(sen_label_test) # 0.5714285714285714
cl.coef_

# We can use the spider/radar chart to display the coefficients of the LinearSVM
# So that we could see the correlations between the face attributes and the election outcomes