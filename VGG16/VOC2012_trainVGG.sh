python3 trainVGG.py \
	--starting_learning_rate=0.00001 \
	--batch_norm_decay=0.997 \
	--gpu_id=0 \
	--resnet_model=vgg_16 \
	--number_of_classes=1000 \
	--output_stride=16 \
	--batch_size=1 \
	2>&1 | tee ./logs/VOC2012VGG16log.txt.`date +'%Y-%m-%d_%H-%M-%S'`
