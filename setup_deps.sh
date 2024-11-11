#!/bin/bash
set -eu

# SETUP DEPS

# Installs Uno dependencies for these tested cases:

# Python    TensorFlow     Platform
#   3.8         2.10.0     Linux & Mac M1
#   3.9         2.13       Linux & Mac M1
#   3.11        2.16       Linux

PV=$( python --version )

# Package tensorflow-gpu goes up to 2.21
# Package tensorflow     continues after that

if [[ $PV == "Python 3.8"* ]]  # Up to TF 2.13
then
  # PROTOBUF="protobuf==3.19.6"
  # TENSORFLOW="tensorflow-gpu==2.10.0"  # Requires CUDA 11.2, not 12.2

  # PROTOBUF="protobuf==3.20.3"
  # TENSORFLOW="tensorflow==2.13.0"        # Requires CUDA 11.8, not 12.2

  PROTOBUF="protobuf==3.20.3"
  TENSORFLOW="tensorflow==2.15.0"        # Requires CUDA 12.2

else  # >= 3.9
  PROTOBUF="protobuf==3.20.3"
  TENSORFLOW="tensorflow==2.16.1"
fi

DEPS=( $PROTOBUF $TENSORFLOW
       "pyarrow==12.0.1"
       pyyaml pandas scikit-learn
     )

set -x
pip install ${DEPS[@]}
