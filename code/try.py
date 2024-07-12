import cv2 as cv
import numpy as np
import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report

# m=np.zeros((100,100,3),dtype=np.uint8)
# m[:,:]=(124,125,126)
# # cv.imshow("m",m)
# h,w,c=m.shape
# B_mean=np.sum(m[:,:,0])/h/w
# G_mean=np.sum(m[:,:,1])/h/w
# R_mean=np.sum(m[:,:,2])/h/w
# print(B_mean,G_mean,R_mean)
# B_std=np.sum((m[:,:,0]-R_mean)**2)
# G_std=m[:,:,1]-G_mean
# R_std=m[:,:,2]-R_mean
# print(B_std)
# cv.waitKey(0)
# cv.destroyAllWindows()

# aa=torch.tensor([2,3,1,5]).cuda()
# bb=torch.tensor([2,3,2,5]).cuda()
# print(aa,bb,(aa==bb).sum().item())
# label=[0,1,1,1,1,0]
# output=[1,1,1,0,0,1]
# TP=2 FN=2 FP=2 TN=0
# M=[2 2;
#    2 0]
# confusion=confusion_matrix(label,output)
# accuracy=accuracy_score(label,output)
# precision=precision_score(label,output,average='binary')
# recall=recall_score(label,output,average='binary')
# f1=f1_score(label,output,average='binary')

# print(confusion)
# print(accuracy,precision,recall,f1,2*precision*recall/(precision+recall))
# print(classification_report(label,output))
# labels=[1,1,1,1,1,2,2,2,2,3,3,3,4,4]
# outputs=[1,1,1,3,3,2,2,3,3,3,4,3,4,3]
#
# confusions=confusion_matrix(labels,outputs)
# reports=classification_report(labels,outputs)
# print(confusions)
# print(reports)
#
# precision_macro=precision_score(labels,outputs,average='macro')
# precision_micro=precision_score(labels,outputs,average='micro')
# precision_weighted=precision_score(labels,outputs,average='weighted')
#
# recall_macro=recall_score(labels,outputs,average='macro')
# recall_micro=recall_score(labels,outputs,average='micro')
# recall_weighted=recall_score(labels,outputs,average='weighted')
#
# f1_macro=f1_score(labels,outputs,average='macro')
# f1_micro=f1_score(labels,outputs,average='micro')
# f1_weighted=f1_score(labels,outputs,average='weighted')
#
# print(precision_macro,precision_micro,precision_weighted)
# print(recall_macro,recall_micro,recall_weighted)
# print(f1_macro,f1_micro,f1_weighted)

a=[0]
print(a)
for i in range(10):
    a.append(i)
    print(a)




