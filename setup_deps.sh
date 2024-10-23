#!/bin/bash
set -eu

# SETUP DEPS

# Installs Uno dependencies for these tested cases:

# Python    TensorFlow     Platform
#    3.8        2.10.0     Linux & Mac M1
#    3.9        2.13       Linux & Mac M1

PV=$( python --version )

if [[ $PV == "Python 3.8"* ]]
then
  PROTOBUF="protobuf==3.19.6"
  TENSORFLOW="tensorflow-gpu==2.10.0"
else  # 3.9
  PROTOBUF="protobuf==3.20.3"
  TENSORFLOW="tensorflow==2.13"
fi

DEPS=( $PROTOBUF $TENSORFLOW
       "pyarrow==12.0.1"
       pyyaml pandas scikit-learn
     )

set -x
pip install ${DEPS[@]}
