#include "OcrLiteImpl.h"
#include "OcrUtils.h"
#include <stdarg.h> //windows&linux

OcrLiteImpl::OcrLiteImpl() {
    loggerBuffer = (char *)malloc(8192);
}

OcrLiteImpl::~OcrLiteImpl() {
    if (isOutputResultTxt) {
        fclose(resultTxt);
    }
    free(loggerBuffer);
}

void OcrLiteImpl::setNumThread(int numOfThread) {
    dbNet.setNumThread(numOfThread);
    angleNet.setNumThread(numOfThread);
    crnnNet.setNumThread(numOfThread);
}

void OcrLiteImpl::initLogger(bool isConsole, bool isPartImg, bool isResultImg) {
    isOutputConsole = isConsole;
    isOutputPartImg = isPartImg;
    isOutputResultImg = isResultImg;
}

void OcrLiteImpl::enableResultTxt(std::string path) {
    isOutputResultTxt = true;
    //printf("resultTxtPath(%s)\n", resultTxtPath.c_str());
    resultTxt = fopen(path.c_str(), "w");
}

void OcrLiteImpl::setGpuIndex(int gpuIndex) {
    dbNet.setGpuIndex(gpuIndex);
    angleNet.setGpuIndex(gpuIndex);
    crnnNet.setGpuIndex(gpuIndex);
}

bool OcrLiteImpl::initModels(const std::string &detPath, const std::string &clsPath,
                         const std::string &recPath, const std::string &keysPath) {
    // Logger("=====Init Models=====\n");
    // Logger("--- Init DbNet ---\n");
    dbNet.initModel(detPath);

    //Logger("--- Init AngleNet ---\n");
    angleNet.initModel(clsPath);

    //Logger("--- Init CrnnNet ---\n");
    crnnNet.initModel(recPath, keysPath);

    //Logger("Init Models Success!\n");
    return true;
}

void OcrLiteImpl::Logger(const char *format, ...) {
    if (!(isOutputConsole || isOutputResultTxt)) return;
    memset(loggerBuffer, 0, 8192);
    va_list args;
    va_start(args, format);
    vsprintf(loggerBuffer, format, args);
    va_end(args);
    if (isOutputConsole) printf("%s", loggerBuffer);
    if (isOutputResultTxt) fprintf(resultTxt, "%s", loggerBuffer);
}

cv::Mat makePadding(cv::Mat &src, const int padding) {
    if (padding <= 0) return src;
    cv::Scalar paddingScalar = {255, 255, 255};
    cv::Mat paddingSrc;
    cv::copyMakeBorder(src, paddingSrc, padding, padding, padding, padding, cv::BORDER_ISOLATED, paddingScalar);
    return paddingSrc;
}

OcrResult OcrLiteImpl::detect(const char *path, const char *imgName,
                          const int padding, const int maxSideLen,
                          float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle) {
    std::string imgFile = getSrcImgFilePath(path, imgName);
    cv::Mat originSrc = imread(imgFile, cv::IMREAD_COLOR);//default : BGR
    int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
    int resize;
    if (maxSideLen <= 0 || maxSideLen > originMaxSide) {
        resize = originMaxSide;
    } else {
        resize = maxSideLen;
    }
    resize += 2 * padding;
    cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
    cv::Mat paddingSrc = makePadding(originSrc, padding);
    ScaleParam scale = getScaleParam(paddingSrc, resize);
    OcrResult result;
    result = detect(path, imgName, paddingSrc, paddingRect, scale,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    return result;
}


OcrResult OcrLiteImpl::detectImageBytes(const uint8_t *data, const long dataLength, const int grey,
                                    const int padding, const int maxSideLen,
                                    float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle,
                                    bool mostAngle) {
    std::vector<uint8_t> vecData(data, data + dataLength);
    cv::Mat originSrc = cv::imdecode(vecData, grey == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);//default : BGR
    OcrResult result;
    result = detect(originSrc, padding, maxSideLen,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    return result;

}

OcrResult OcrLiteImpl::detectBitmap(uint8_t *bitmapData, int width, int height, int channels, int padding,
                                int maxSideLen, float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle,
                                bool mostAngle) {
    cv::Mat originSrc(height, width, CV_8UC(channels), bitmapData);
    if (channels > 3) {
        cv::cvtColor(originSrc, originSrc, cv::COLOR_RGBA2BGR);
    } else if (channels == 3) {
        cv::cvtColor(originSrc, originSrc, cv::COLOR_RGB2BGR);
    }
    OcrResult result;
    result = detect(originSrc, padding, maxSideLen,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    return result;
}


OcrResult OcrLiteImpl::detect(const cv::Mat &mat, int padding, int maxSideLen, float boxScoreThresh, float boxThresh,
                          float unClipRatio, bool doAngle, bool mostAngle) {
    cv::Mat originSrc = mat;
    int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
    int resize;
    if (maxSideLen <= 0 || maxSideLen > originMaxSide) {
        resize = originMaxSide;
    } else {
        resize = maxSideLen;
    }
    resize += 2 * padding;
    cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
    cv::Mat paddingSrc = makePadding(originSrc, padding);
    ScaleParam scale = getScaleParam(paddingSrc, resize);
    OcrResult result;
    result = detect(NULL, NULL, paddingSrc, paddingRect, scale,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    return result;
}

std::vector<cv::Mat> OcrLiteImpl::getPartImages(cv::Mat &src, std::vector<TextBox> &textBoxes,
                                            const char *path, const char *imgName) {
    std::vector<cv::Mat> partImages;
    for (size_t i = 0; i < textBoxes.size(); ++i) {
        cv::Mat partImg = getRotateCropImage(src, textBoxes[i].boxPoint);
        partImages.emplace_back(partImg);
        //OutPut DebugImg
        if (isOutputPartImg) {
            std::string debugImgFile = getDebugImgFilePath(path, imgName, i, "-part-");
            saveImg(partImg, debugImgFile.c_str());
        }
    }
    return partImages;
}

OcrResult OcrLiteImpl::detect(const char *path, const char *imgName,
                          cv::Mat &src, cv::Rect &originRect, ScaleParam &scale,
                          float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle) {

    cv::Mat textBoxPaddingImg = src.clone();
    int thickness = getThickness(src);

    Logger("  \"scale_params\": {\n");
    Logger("    \"original_width\": %d,\n", scale.srcWidth);
    Logger("    \"original_height\": %d,\n", scale.srcHeight);
    Logger("    \"target_width\": %d,\n", scale.dstWidth);
    Logger("    \"target_height\": %d,\n", scale.dstHeight);
    Logger("    \"width_ratio\": %f,\n", scale.ratioWidth);
    Logger("    \"height_ratio\": %f\n", scale.ratioHeight);
    Logger("  },\n");
    Logger("  \"detection_result\": {\n");
    Logger("    \"bounding_boxes\": {\n");
    double startTime = getCurrentTime();
    std::vector<TextBox> textBoxes = dbNet.getTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio);
    double endDbNetTime = getCurrentTime();
    double dbNetTime = endDbNetTime - startTime;
    Logger("      \"elapsed_time\": %f,\n", dbNetTime/1000);
    Logger("      \"result\": [\n");

    for (size_t i = 0; i < textBoxes.size(); ++i) {
        std::string tailComma = (i == textBoxes.size() - 1) ? "\n" : ",\n";
        Logger("        {\n");
        Logger("          \"score\": %f,\n", textBoxes[i].score);
        Logger("          \"top_left\": {\n", textBoxes[i].score);
        Logger("            \"x\": %d,\n", textBoxes[i].boxPoint[0].x);
        Logger("            \"y\": %d\n", textBoxes[i].boxPoint[0].y);
        Logger("          },\n");
        Logger("          \"top_right\": {\n");
        Logger("            \"x\": %d,\n", textBoxes[i].boxPoint[1].x);
        Logger("            \"y\": %d\n", textBoxes[i].boxPoint[1].y);
        Logger("          },\n");
        Logger("          \"bottom_right\": {\n");
        Logger("            \"x\": %d,\n", textBoxes[i].boxPoint[2].x);
        Logger("            \"y\": %d\n", textBoxes[i].boxPoint[2].y);
        Logger("          },\n");
        Logger("          \"bottom_left\": {\n");
        Logger("            \"x\": %d,\n", textBoxes[i].boxPoint[3].x);
        Logger("            \"y\": %d\n", textBoxes[i].boxPoint[3].y);
        Logger("          }\n");
        Logger("        }%s",tailComma.c_str());
    }
    Logger("      ]\n");
    Logger("    },\n");

    // Logger("---------- step: drawTextBoxes ----------\n");
    //drawTextBoxes(textBoxPaddingImg, textBoxes, thickness);

    //---------- getPartImages ----------
    std::vector<cv::Mat> partImages = getPartImages(src, textBoxes, path, imgName);

    //Logger("---------- step: angleNet getAngles ----------\n");
    Logger("    \"text_orientation\": {\n");
    Logger("      \"result\": [\n");
    std::vector<Angle> angles;
    double start = getCurrentTime();
    angles = angleNet.getAngles(partImages, path, imgName, doAngle, mostAngle);
    double end = getCurrentTime();
    float totalOrientationTime = 0.0f;
    for (size_t i = 0; i < angles.size(); ++i) {
        std::string tailComma = (i == textBoxes.size() - 1) ? "\n" : ",\n";
        Logger("        {\n");
        Logger("          \"index\": %d,\n", angles[i].index);
        Logger("          \"score\": %f\n", angles[i].score);
        Logger("        }%s", tailComma.c_str());
        totalOrientationTime += angles[i].time;
    }
    Logger("      ],\n");
    Logger("      \"elapsed_time\": %f\n", (end-start)/1000);
    Logger("    },\n");

    //Rotate partImgs
    for (size_t i = 0; i < partImages.size(); ++i) {
        if (angles[i].index == 1) {
            partImages.at(i) = matRotateClockWise180(partImages[i]);
        }
    }

    Logger("    \"text_recognition\": {\n");
    Logger("      \"result\": [\n");
    start = getCurrentTime();
    std::vector<TextLine> textLines = crnnNet.getTextLines(partImages, path, imgName);
    end = getCurrentTime();

    for (size_t i = 0; i < textLines.size(); ++i) {
        std::string tailComma = (i == textLines.size() - 1) ? "\n" : ",\n";
        Logger("        {\n");
        Logger("          \"text\": \"%s\",\n", escapeJsonString(textLines[i].text.c_str()).c_str());
        Logger("          \"char_scores\": [");
        for (size_t s = 0; s < textLines[i].charScores.size(); ++s) {
            if (s > 0) {
                Logger(", ");
            }
            Logger("%f", textLines[i].charScores[s]);
        }
        Logger("]\n");
        Logger("        }%s", tailComma.c_str());
    }
    Logger("      ],\n");
    Logger("      \"elapsed_time\": %f\n", (end-start)/1000);
    Logger("    }\n");

    std::vector<TextBlock> textBlocks;
    for (size_t i = 0; i < textLines.size(); ++i) {
        std::vector<cv::Point> boxPoint = std::vector<cv::Point>(4);
        int padding = originRect.x;//padding conversion
        boxPoint[0] = cv::Point(textBoxes[i].boxPoint[0].x - padding, textBoxes[i].boxPoint[0].y - padding);
        boxPoint[1] = cv::Point(textBoxes[i].boxPoint[1].x - padding, textBoxes[i].boxPoint[1].y - padding);
        boxPoint[2] = cv::Point(textBoxes[i].boxPoint[2].x - padding, textBoxes[i].boxPoint[2].y - padding);
        boxPoint[3] = cv::Point(textBoxes[i].boxPoint[3].x - padding, textBoxes[i].boxPoint[3].y - padding);
        TextBlock textBlock{boxPoint, textBoxes[i].score, angles[i].index, angles[i].score,
                            angles[i].time, textLines[i].text, textLines[i].charScores, textLines[i].time,
                            angles[i].time + textLines[i].time};
        textBlocks.emplace_back(textBlock);
    }

    double endTime = getCurrentTime();
    double fullTime = endTime - startTime;
    Logger("  },\n");
    Logger("  \"total_elapsed_time\": %f,\n", fullTime/1000);
    

    //cropped to original size
    cv::Mat textBoxImg;

    if (originRect.x > 0 && originRect.y > 0) {
        textBoxPaddingImg(originRect).copyTo(textBoxImg);
    } else {
        textBoxImg = textBoxPaddingImg;
    }

    //Save result.jpg
    if (isOutputResultImg) {
        std::string resultImgFile = getResultImgFilePath(path, imgName);
        imwrite(resultImgFile, textBoxImg);
    }

    std::string strRes;
    for (auto &textBlock: textBlocks) {
        strRes.append(textBlock.text);
        strRes.append("\n");
    }

    return OcrResult{dbNetTime, textBlocks, textBoxImg, fullTime, strRes};
}
