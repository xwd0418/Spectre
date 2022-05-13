cd /workspace/smart4.0
mkdir -p /workspace/smart4.0/tempdata
cp /data/data/data_cleaned.zip /workspace/smart4.0/tempdata
cp /data/data/specs_peaks.zip /workspace/smart4.0/tempdata
unzip -q ./tempdata/data_cleaned.zip -d ./tempdata
unzip -q ./tempdata/specs_peaks.zip -d ./tempdata
echo "Done Copying Pair Data (cp_pair_data.sh)"