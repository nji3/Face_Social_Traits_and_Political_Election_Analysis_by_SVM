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
hog_dat = np.column_stack((all_landmarks,Hogs)) # 491x6432

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
gov_im = '*/data/img-elec/governor/*.jpg'
sen_im = '*/data/img-elec/senator/*.jpg'
gov_hog = extract_hog_features(gov_im)
sen_hog = extract_hog_features(sen_im)
gov_hog = min_max(gov_hog)
sen_hog = min_max(sen_hog)

# Combine landmarks and hog features
gov_dat = np.column_stack((gov_landmarks,gov_hog)) # 112x6432
sen_dat = np.column_stack((sen_landmarks,sen_hog)) # 116x6432

## Task 2.2
# Two layers of SVM
# First layer contains 14 models for 14 traits
# Read the selected 14 models
svr_res = np.load('1.2SVR_Res.npy')
svr_params = svr_res[0]

# For governors
pred_trait = []
for i in range(14):
    clf = SVR(C=svr_params[i][0],gamma=svr_params[i][1],epsilon=svr_params[i][2])
    clf.fit(hog_dat,all_traits_value[:,i])
    pred_trait.append(clf.predict(gov_dat))
gov_traits = np.array(pred_trait).T
cor_coef = []
for i in range(14):
    cor_coef.append(np.corrcoef(np.vstack((gov_traits[:,i],gov_vote_diff[:,0]))))
gov_pair_traits,gov_pair_label = dat_rankSVM(gov_traits,gov_vote_diff)
gov_pair_train,gov_pair_test,gov_label_train,gov_label_test=train_test_split(gov_pair_traits,\
gov_pair_label, test_size=0.3, random_state=231)
C = np.linspace(30,60,15)
k = ShuffleSplit(n_splits=5,random_state=231)
clf = GridSearchCV(estimator=LinearSVC(C=1,fit_intercept=False),\
param_grid=dict(C=C),cv=k,n_jobs=-1)
clf.fit(gov_pair_train,gov_label_train)
clf.best_score_ # 0.7
clf.best_estimator_.C # 32.14
cl = LinearSVC(C=clf.best_estimator_.C,fit_intercept=False)
cl.fit(gov_pair_train,gov_label_train)
cl.score(gov_pair_train,gov_label_train) # 0.8205128205128205
sum(cl.predict(gov_pair_test)==gov_label_test)/len(gov_label_test) # 0.5882352941176471
cl.coef_
cl.score()

# For senators
pred_trait = []
for i in range(14):
    clf = SVR(C=svr_params[i][0],gamma=svr_params[i][1],epsilon=svr_params[i][2])
    clf.fit(hog_dat,all_traits_value[:,i])
    pred_trait.append(clf.predict(sen_dat))
sen_traits = np.array(pred_trait).T
cor_coef = []
for i in range(14):
    cor_coef.append(np.corrcoef(np.vstack((sen_traits[:,i],sen_vote_diff[:,0]))))
sen_pair_traits,sen_pair_label = dat_rankSVM(sen_traits,sen_vote_diff)
sen_pair_train,sen_pair_test,sen_label_train,sen_label_test=train_test_split(sen_pair_traits,\
sen_pair_label, test_size=0.3, random_state=231)
C = np.linspace(10,50,10)
k = ShuffleSplit(n_splits=5,random_state=231)
clf = GridSearchCV(estimator=LinearSVC(C=1,fit_intercept=False),\
param_grid=dict(C=C),cv=k,n_jobs=-1)
clf.fit(sen_pair_train,sen_label_train)
clf.best_score_ # 0.575
clf.best_estimator_.C # 32.22
cl = LinearSVC(C=clf.best_estimator_.C,fit_intercept=False)
cl.fit(sen_pair_train,sen_label_train)
cl.score(sen_pair_train,sen_label_train) # 0.759493670886076
sum(cl.predict(sen_pair_test)==sen_label_test)/len(sen_label_test) # 0.6285714285714286
cl.coef_

# We can use the spider/radar chart to display the coefficients of the LinearSVM
# So that we could see the correlations between the face attributes and the election outcomes