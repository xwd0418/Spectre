tempdata="/workspace/smart4.5/tempdata"
mkdir -p $tempdata

select FILE in $(ls /data/data | grep zip)
do
  trimmed=${FILE%.zip}
  echo "======"
  read -p "Pull $trimmed? [Y/N]:"$'\n    > ' -n 1 -r
  echo    # (optional) move to a new line
  if [[ ! $REPLY =~ ^[Yy]$ ]]
  then
    echo "Aborted Pulling $trimmed"
    exit 1
  fi
  echo "Pulling $trimmed..."
  
  start=`date +%s`
  cp "/data/data/$FILE" $tempdata
  unzip -q "$tempdata/$FILE" -d $tempdata
  rm "$tempdata/$FILE"
  end=`date +%s`
  runtime=`expr $end - $start`

  echo "Done Pulling $trimmed!!!"
  echo "Time taken: $runtime (s)"

  exit
done