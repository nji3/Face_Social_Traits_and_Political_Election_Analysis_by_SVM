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
gov_im = '*/data/img-elec/governor/*.jpg'
sen_im = '*/data/img-elec/senator/*.jpg'
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
k = ShuffleSplit(n_splits=5,random_state=231)
clf = GridSearchCV(estimator=SVC(C=1), param_grid=dict(C=C,gamma=Gamma),cv=k,n_jobs=-1)
clf.fit(gov_new_dat, gov_label)
clf.best_score_ # 0.7333333333333333
clf.best_estimator_ # C = 5, gamma = 0.01
cl = SVC(C=clf.best_estimator_.C,gamma=clf.best_estimator_.gamma)
scores = cross_val_score(cl, gov_new_dat,gov_label, cv=k, scoring='precision')
scores.mean() # 0.7325396825396826

# Do K-fold cross validation to find the Parameters we want
sen_new,sen_label = dat_rankSVM(sen_dat,sen_vote_diff)
C = np.linspace(1,5,5)
Gamma = np.linspace(0.001,0.01,5)
k = ShuffleSplit(n_splits=5,random_state=231)
clf = GridSearchCV(estimator=SVC(C=1), param_grid=dict(C=C,gamma=Gamma),cv=k,n_jobs=-1)
clf.fit(sen_new,sen_label)
clf.best_score_ # 0.8
clf.best_estimator_ # C = 5, gamma = 0.00325
cl = SVC(C=clf.best_estimator_.C,gamma=clf.best_estimator_.gamma)
scores = cross_val_score(cl, sen_new, sen_label, cv=k, scoring='precision')
scores.mean() # 0.825873015873016