export DATA_VERSION=0_ori

export DATA_DIR=/mnt/e/Workspace/sky/data/$DATA_VERSION

export BOX_DIR=$DATA_VERSION/box

mkdir -p $BOX_DIR
cp -r $DATA_DIR/Camera/rgb $BOX_DIR
cp -r $DATA_DIR/Camera/object_detection $BOX_DIR

cd $DATA_VERSION

zip box.zip box