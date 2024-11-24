#!/usr/bin/env bash

echo "========Compilation Options========"
echo "Please choose the compilation option:"
echo "1) Release (default)"
echo "2) Debug"
read -p "" BUILD_TYPE
if [ $BUILD_TYPE == 1 ]; then
  BUILD_TYPE=Release
elif [ $BUILD_TYPE == 2 ]; then
  BUILD_TYPE=Debug
else
  BUILD_TYPE=Release
fi

BUILD_OUTPUT="BIN"

echo "Choose ONNXRuntime destination"
echo "1) CPU (default)"
echo "2) CUDA"
echo "Note: The example project is integrated with the CPU version by default, the CUDA version only supports Linux64 and needs to be downloaded."
read -p "" ONNX_TYPE
if [ $ONNX_TYPE == 1 ]; then
  ONNX_TYPE="CPU"
elif [ $ONNX_TYPE == 2 ]; then
  ONNX_TYPE="CUDA"
else
  ONNX_TYPE="CPU"
fi

sysOS=$(uname -s)
NUM_THREADS=1
if [ $sysOS == "Darwin" ]; then
  NUM_THREADS=$(sysctl -n hw.ncpu)
elif [ $sysOS == "Linux" ]; then
  NUM_THREADS=$(grep ^processor /proc/cpuinfo | wc -l)
else
  echo "Cannot identify your OS: $sysOS"
fi

mkdir -p $sysOS-$ONNX_TYPE-$BUILD_OUTPUT
pushd $sysOS-$ONNX_TYPE-$BUILD_OUTPUT

echo "cmake -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DOCR_OUTPUT=$BUILD_OUTPUT -DOCR_ONNX=$ONNX_TYPE .."
cmake -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DOCR_OUTPUT=$BUILD_OUTPUT -DOCR_ONNX=$ONNX_TYPE ..
cmake --build . --config $BUILD_TYPE -j $NUM_THREADS
cmake --build . --config $BUILD_TYPE --target install
popd
