import numpy as np
import os
import json
import cv2 as cv
prep_path = '..\\prep\\'
preproc_name = 'Preprocess.exe'
folder_path = '..\\pic\\'
train_path = '..\\ReID\\train.py'
test_path = '..\\ReID\\test.py'
prep_cmd = prep_path+preproc_name+' '+folder_path


def execCmd(cmd):
	r = os.popen(cmd)
	text = r.read()
	r.close()
	return text

def track(strJson,pic_id,lastJSON):
	data = json.loads(strJson)
	num = data["rect_num"]
	for i in range(0,num):
		if lastJSON:
			minDis = 999999
			label = 0
			for j in range(0,num):
				if (data["rect"][i][0]-lastJSON["rect"][j][0])**2+(data["rect"][i][1]-lastJSON["rect"][j][1])**2<minDis:
					minDis = (data["rect"][i][0]-lastJSON["rect"][j][0])**2+(data["rect"][i][1]-lastJSON["rect"][j][1])**2
					label = j
			os.system('mv '+'../pic/'+str(i)+'/'+str(pic_id)+'.jpg '+'../train/'+str(label))
		else:
			os.system('mkdir ..\\train\\'+str(i))
			os.system('mv '+'../pic/'+str(i)+'/'+str(pic_id)+'.jpg '+'../train/'+str(i))
		
	return data.copy()

def toDisk(img, pic_id):
	cv.imwrite(folder_path+str(pic_id)+'.jpg',img)

def train():
	pic_id =0
	img = 0
	lastJSON = {}
	while(True):
		img = (yield img)
		if(isinstance(img,int)):
			os.system('mv -f '+'../train'+' '+'../Market/pytorch/')
			os.system('python '+ train_path)
			os.system('rm -rf ../Market/pytorch/train')
			break
		pic_id = pic_id+1
		toDisk(img,pic_id)
		text = execCmd(prep_cmd+' '+str(pic_id)+'.jpg')
		lastJSON = track(text,pic_id,lastJSON)



def test():
	img = 0
	pic_id = 0
	while(True):
		img = (yield img)
		if(isinstance(img,int)):
			return ' ..\\json\\'+str(pic_id)+'.json'
		pic_id = pic_id+1
		toDisk(img,pic_id)
		text = execCmd(prep_cmd+' '+str(pic_id)+'.jpg test')
		with open('..\\json\\'+str(pic_id)+'.json','w') as f:
			json.dump(json.loads(text),f)
		dataj = json.loads(text)
		for i in range(dataj['rect_num']):
			os.system('mv -f '+'../train/'+str(i)+' '+'../Market/pytorch/query')
		
		os.system('python '+test_path+' --json ..\\json\\'+str(pic_id)+'.json')

