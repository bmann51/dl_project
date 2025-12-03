## For coco you need to
### Set a basepath
mkdir -p ~/data/coco/images ~/data/coco96
cd ~/data/coco/images

### Download Coco Images
curl -O http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

### Use the resize_coco_to_96.py file
