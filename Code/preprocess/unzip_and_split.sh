#!/bin/bash

# Usage: bash preprocess/unzip_and_split.sh /home/ubuntu/directory_with_jester_data

# This script assumes that all 23 .tgz files, jester-v1-train.csv and jester-v1-validation.csv are in the directory passed as an argument to this script
# Make sure you have around 40 GB of disk space available before running this!

DATA_DIR=$1
echo "Data directory: ${DATA_DIR}"

pushd ${DATA_DIR} > /dev/null

#echo "Unzipping .tgz files"
#cat data_*.tgz | tar -xzvf -

mkdir train_imgseqs test_imgseqs notfound_imgseqs

NOT_FOUND_COUNT=0

echo "Splitting between train and test"
for IMGSEQ_ID in $( ls 20bn-jester-v1 )
do
    if grep -q "^${IMGSEQ_ID};" jester-v1-train.csv
    then
        echo "${IMGSEQ_ID}: Found in jester-v1-train.csv"
        mv 20bn-jester-v1/${IMGSEQ_ID} train_imgseqs/
    elif grep -q "^${IMGSEQ_ID};" jester-v1-validation.csv
    then
        echo "${IMGSEQ_ID}: Found in jester-v1-validation.csv"
        mv 20bn-jester-v1/${IMGSEQ_ID} test_imgseqs/
    else
        echo "${IMGSEQ_ID}: Not found in jester-v1-train.csv or jester-v1-validation.csv!"
        mv 20bn-jester-v1/${IMGSEQ_ID} notfound_imgseqs/
        NOT_FOUND_COUNT=$((NOT_FOUND_COUNT+1))
    fi
done

echo "${NOT_FOUND_COUNT} image sequences were not found in train jester-v1-train.csv or jester-v1-validation.csv"
echo "They have been moved to the notfound_imgseqs directory"

popd ${DATA_DIR} > /dev/null