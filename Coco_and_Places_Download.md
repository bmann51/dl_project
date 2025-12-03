## For coco you need to
### Set a basepath
mkdir -p ~/data/coco/images ~/data/coco96 \
cd ~/data/coco/images

### Download Coco Images
curl -O http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

### Use the resize_coco_to_96.py file

## For Places you need to
### Set a basepath
mkdir -p ~/data/places365/raw ~/data/places365/96x96 \
cd ~/data/places365/raw

### Download Places data
tar -xf places365standard_easyformat.tar -C ~/data/places365/raw \
note: there are other download commands

### Use the resize_places_to_96.py file
