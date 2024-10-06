# fastwork-images-classification-yolov8

## prepare data
```
data_raw--class1--1.png
                  --2.png
        --class2--1.png
                  --2.png
```
## run for split/train/evaualtion
```
#split data_raw to data[train,val,test]
python 0_split_images.py

#training
python 1_train.py

#evaluation
python 2_run_evaluation.py
```
## copy to streamlit app
```
1. copy runs/classify/train{x} to evaluations floder
2. copy runs/classify/train{x}/weights/best.pt
take 1,2 to streamlit app
```
## streamlit-app
```
page--evaluations
    --example_code
app.py
images--1.png
      --2.png
best.pt
evaluations
sample_run_predict.py
packages.txt
requirements.txt

```
## setup github for upload large files
```
git clone https://github.com/phawitcrma/rice-disease-app.git
cd rice-disease-app/
git lfs track "*.pt"
git add .
git commit -m "LFS"
git remote set-url origin https://xxxxxxxx@github.com/phawitcrma/rice-disease-app
git push origin main

#if error occure
sudo apt update
sudo apt install git-lfs
git config --global user.email "phawit.bo@crma.ac.th"
git config --global user.name "phawitcrma"

## how to get token

https://www.youtube.com/watch?v=ePCBuIQJAUc
```

## Already Deploy
https://github.com/phawitcrma/rice-disease-app.git
https://rice-disease-app-flginxk7cvdnaqbxbhg64w.streamlit.app/


