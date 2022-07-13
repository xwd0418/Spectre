#!/bin/bash

tempdata=/workspace/smart4.5/tempdata
mkdir -p $tempdata

# pulls a certain dataset and unzips it
FILE="/data/data/$1"
if [ -f "$FILE" ]; then
    echo "Loading $FILE"
    cp $FILE $tempdata
    unzip -q "$tempdata/$1" -d $tempdata
    echo "Done Copying $FILE"
    exit 0
else
    exit 1
fi