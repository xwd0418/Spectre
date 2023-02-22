#!/bin/bash

tempdata=/workspace/smart4.5/tempdata
mkdir -p $tempdata

# pulls a certain dataset and unzips it
FILE="/data/data/$1"
if [ -f "$FILE.zip" ]; then
    start=`date +%s`

    echo "Loading $FILE.zip"
    cp "$FILE.zip" $tempdata
    unzip -q "$tempdata/$1.zip" -d $tempdata
    echo "Done Copying $FILE"
    exit 0

    end=`date +%s`
    runtime=`expr $end - $start`
    echo "Time taken: $runtime (s)"
else
    echo "File $FILE(.zip) does not exist in /data/data"
    exit 1
fi