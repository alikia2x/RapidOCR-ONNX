# RapidOCR-ONNX

## Project Introduction

RapidOCR-ONNX is an efficient Optical Character Recognition (OCR) tool that runs [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) based on [ONNX-Runtime](https://github.com/microsoft/onnxruntime/). This project aims to achieve cross-platform efficient inference and provides output in JSON format for easy integration and processing by other programs.

## Features

- Efficient OCR inference using ONNX-Runtime.
- Supports multiple platforms to ensure broad compatibility.
- Outputs results in JSON format, making it easy to integrate and use.

## Build

We have tested the build on macOS.
Build instructions for Windows and Linux will be available soon.

You can also go to the release page to download the executable files we built. Currently, Release is also only available for macOS.

### Preparations

First, download the pre-built versions of ONNX Runtime and OpenCV from [our Gitea repository](https://git.alikia2x.com/alikia2x/openrewind-assets/releases).

For backup purposes, you can also download them from the following GitHub repositories:

- [OpenCVBuilder](https://github.com/RapidAI/OpenCVBuilder/releases)  
- [ONNX Runtime Builder](https://github.com/RapidAI/OnnxruntimeBuilder/releases)

Please note that the compilation results provided by the ONNX Runtime Builder repository may have an incorrect directory structure. You can refer to the files available in our Gitea repository to make the necessary corrections.

After downloading, place the unzipped folder (named "macos") into the corresponding directories in the project root: `onnxruntime-static` and `opencv-static`.

### Build on macOS

```bash
chmod +x build.sh
./build.sh
```

Tap Enter twice, you'll get the results in `Darwin-CPU-BIN`.

## Usage

Run the following command in the command line to use RapidOCR-ONNX:

```bash
./RapidOcrOnnx (-d --models) (-1 --det) (-2 --cls) (-3 --rec) (-4 --keys) (-i --image)
[-t --numThread] [-p --padding] [-s --maxSideLen]
[-b --boxScoreThresh] [-B --boxThresh] [-u --unClipRatio]
[-a --doAngle] [-A --mostAngle] [-G --GPU] [-o --output]
```

### Required Parameters
- `-d --models`: Model directory.
- `-1 --det`: Detection model filename.
- `-2 --cls`: Classification model filename.
- `-3 --rec`: Recognition model filename.
- `-4 --keys`: Key filename.
- `-i --image`: Path to the target image.
- `-o --output`: Path for the output results in JSON format.

### Optional Parameters
- `-t --numThread`: Number of threads (integer), default: 4.
- `-p --padding`: Padding value (integer), default: 50.
- `-s --maxSideLen`: Maximum side length of the image (integer), default: 1024.
- `-b --boxScoreThresh`: Box score threshold (float), default: 0.5.
- `-B --boxThresh`: Box threshold (float), default: 0.3.
- `-u --unClipRatio`: Unclipping ratio (float), default: 1.6.
- `-a --doAngle`: Enable (1) / Disable (0) angle network, default: enabled.
- `-A --mostAngle`: Enable (1) / Disable (0) most likely angle index, default: enabled.
- `-G --GPU`: Disable (-1) / GPU0 (0) / GPU1 (1) / ... Use Vulkan GPU acceleration, default: disabled (-1).

### Other Parameters
- `-v --version`: Display version information.
- `-h --help`: Print help information.

## Examples
Here are example commands for using RapidOCR-ONNX:

```bash
# Example 1
./RapidOcrOnnx --models models --det det.onnx --cls cls.onnx --rec rec.onnx --keys keys.txt --image 1.jpg --GPU 0 --output 1.json

# Example 2
./RapidOcrOnnx -d models -1 det.onnx -2 cls.onnx -3 rec.onnx -4 keys.txt -i 1.jpg -t 4 -p 50 -s 0 -b 0.6 -B 0.3 -u 2.0 -a 1 -A 1 -G 0 -o 1.json
```

## Output Format
The output will be
 returned in JSON format, as shown below:

```json
{
  "input_params": {
    "threads": 1,
    "padding": 50,
    "longest_side": 1024,
    "box_score_threshold": 0.5,
    "box_threshold": 0.3,
    "detect_orientation": 1,
    "auto_rotate": 1,
    "GPU": -1
  },
  "scale_params": {
    "original_width": 3520,
    "original_height": 2324,
    "target_width": 1120,
    "target_height": 736
  },
  "detection_result": {
    "bounding_boxes": {
      "elapsed_time": 0.187204,
      "result": [
        {
          "score": 0.596514,
          "top_left": { "x": 74, "y": 66 },
          "bottom_right": { "x": 283, "y": 113 }
        },
        {
          "score": 0.621396,
          "top_left": { "x": 839, "y": 66 },
          "bottom_right": { "x": 1298, "y": 116 }
        }
      ]
    },
    "text_orientation": {
      "result": [
        {
          "index": 0,
          "score": 0.999959
        },
        {
          "index": 0,
          "score": 0.993791
        }
      ],
      "elapsed_time": 0.038303
    },
    "text_recognition": {
      "result": [
        {
          "text": "Text",
          "char_scores": [0.057412, 0.986533, 0.999718, 0.71237]
        },
        {
          "text": "你好",
          "char_scores": [0.98043, 0.996933]
        }
      ],
      "elapsed_time": 0.038303
    }
  },
  "total_elapsed_time": 0.973727,
  "text": "Text\n你好"
}
```

## Contribution
Contributions of any form are welcome! If you find any bugs or have suggestions for improvements, please submit an issue or a pull request.

## License
This project is licensed under the Apache 2.0 License. For more details, please refer to the LICENSE file.

## Contact
For more information or support, please contact the project maintainer or submit an issue on GitHub.

Thank you for using RapidOCR-ONNX! We hope this tool helps you efficiently perform optical character recognition.
