import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../examples/pca_alex/deploy.prototxt'  				#deploy.prototxt
PRETRAINED = '../examples/pca_alex/caffe_pca_alex_train_iter_4500_b.caffemodel'	#Caffemodel
IMAGE_FILE = '/mnt/share/ILSVRC2015/pca2/ILSVRC2015_train_00013004_000022.jpg'	##class 2				#Image to test
#IMAGE_FILE = '/mnt/share/ILSVRC2015/pca1/ILSVRC2015_train_00184001_000000.jpg'      ##class 1
##IMAGE_FILE = '/mnt/share/ILSVRC2015/pca/ILSVRC2015_train_00030002_000000.jpg'      ##class 0
CATEGORIES = '../examples/pca_alex/category.txt'				#Image label categories

caffe.set_mode_cpu()											#set mode to CPU
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.array([105.908874512,114.063842773,116.282836914]),	#mean value for each channel 
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(128, 128))										#Image size

input_image = caffe.io.load_image(IMAGE_FILE)										#Read Image
plt.imshow(input_image)

prediction = net.predict([input_image])  
## predict takes any number of images, and formats them for the Caffe net automatically
##print 'prediction shape:', prediction[0].shape
# plt.plot(prediction[0])
##print 'predicted class:', prediction[0].argmax() ##get the top 1 prediction
labels = np.genfromtxt(CATEGORIES,delimiter=' ',dtype=None,names=('label','number'))

def top2(a):
	return np.argsort(a)[::-1][:2]

toptwo = top2(prediction[0])									#in my case I use top 5 prediction

print 'prediction       labels'
for i in toptwo:
	print round(prediction[0][i],4),'        	',labels[i][0]
##plt.show()
