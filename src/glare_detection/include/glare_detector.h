#ifndef GLARE_DETECTOR_H
#define GLARE_DETECTOR_H

extern int debug_color;

#include <opencv2/opencv.hpp>

class glare_detector {
private:
    cv::Mat currentFrame;
    cv::Point glareCenter;
    double detectedArea;
    bool glareFound;
    
    // Local Contrast 계산: 17x17 블록 단위로 RMS 대비 계산
    cv::Mat computeLocalContrast(const cv::Mat& intensity);

public:
    glare_detector();
    void startVideo(const cv::Mat& frame);
    void endVideo();

    // Photometric feature map 계산:
    // HSV의 intensity(V), saturation(S), local contrast를 기반으로 계산.
    cv::Mat computePhotometricMap(const cv::Mat& inputRGB);
    
    // Geometric feature map 계산:
    // Gphoto map에 Gphoto map에서 원형률 조건을 만족하는 영역을 찾아 ggeo map으로 반환
    cv::Mat computeGeometricMap(const cv::Mat& gphoto);
    
    cv::Mat combineMaps(const cv::Mat& gphoto, const cv::Mat& ggeo);
    cv::Mat computePriorityMap(const cv::Mat& gphoto, const cv::Mat& ggeo);
    double getDetectedArea() const;
    void drawGlareContours(const cv::Mat& inputImage, cv::Mat& frame);

    double isBrightArea(const cv::Mat& frame);
    double isStandardArea(const cv::Mat& frame);

};

#endif