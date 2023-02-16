select PATH1 in $(ls /data/smart4.5)
do
  select PATH2 in $(ls /data/smart4.5/$PATH1)
  do
    TB_PATH="/data/smart4.5/$PATH1/$PATH2"
    echo "Launching on path $TB_PATH"
    tensorboard --logdir $TB_PATH
    exit
  done
  exit
done