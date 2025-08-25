// 본 코드는 Glare Detection 성능 평가만을 위해 따로 만들어진 것으로, 빌드 파일에는 속하지 않음.
// 빌드 환경

// (프로젝트 루트)
// ├─ include/
// │  ├─ glare_detector.h
// │  └─ return_position.h
// ├─ src/
// │  ├─ gt_labels/        # GT 라벨 파일 폴더
// │  ├─ test_images/      # 평가용 이미지 폴더
// │  ├─ glare_detector.cpp
// │  ├─ glare_evaluate.cpp
// │  ├─ main.cpp
// │  └─ return_position.cpp
// └─ glare_evaluate       # 빌드 산출물

// 빌드 방법 : $ g++ -std=c++17     src/glare_evaluate.cpp     src/glare_detector.cpp     src/return_position.cpp     -Iinclude     `pkg-config --cflags --libs opencv4`     -o glare_evaluate

// glare_evaluate.cpp (with resizing applied + arrow key navigation)
// - Priority 1/2를 동일한 "Glare"로 시각화 (파란색 박스)
// - Ground Truth는 빨간색 박스
// - Legend는 "Ground Truth", "Prediction (Glare)" 두 항목만 표시
// - 좌/우 화살표로 이미지 결과 탐색, ESC로 종료

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <map>
#include "glare_detector.h"
#include "return_position.h"

namespace fs = std::filesystem;

// === 결과 시각화를 위한 리사이즈 해상도 ===
#define RESIZED_WIDTH 640
#define RESIZED_HEIGHT 480

// YOLO txt 라벨 파싱용 구조체 (정규화된 형식)
struct YoloLabel {
    float cx, cy, w, h; // normalized (0~1)
};

// YOLO 라벨 -> OpenCV Rect 변환 (원본 이미지 크기 기준)
cv::Rect2f yoloToRect(const YoloLabel& label, int img_w, int img_h) {
    float x = (label.cx - label.w / 2.0f) * img_w;
    float y = (label.cy - label.h / 2.0f) * img_h;
    float w = label.w * img_w;
    float h = label.h * img_h;
    return cv::Rect2f(x, y, w, h);
}

// IoU 계산 (교집합/합집합)
float computeIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    float interArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float unionArea = a.area() + b.area() - interArea;
    return (unionArea > 0.0f) ? interArea / unionArea : 0.0f;
}

// YOLO 라벨 파일(.txt) 로드 후 Rect 리스트로 변환 (원본 이미지 크기 기준)
std::vector<cv::Rect2f> loadYoloLabels(const std::string& label_path, int img_w, int img_h) {
    std::ifstream in(label_path);
    std::vector<cv::Rect2f> boxes;
    if (!in.is_open()) return boxes;

    int class_id;
    YoloLabel lbl;
    while (in >> class_id >> lbl.cx >> lbl.cy >> lbl.w >> lbl.h) {
        boxes.push_back(yoloToRect(lbl, img_w, img_h));
    }
    return boxes;
}

// 우선순위 맵(priority)만 사용해 예측 박스 추출
// - 내부 로직: Priority 1 먼저 시도, 없으면 Priority 2 시도 (우선순위 유지)
// - 시각화는 Priority 구분 없이 "Glare"로 통합(파란색 박스)
std::pair<cv::Rect2f, int> getPredictedBox(const cv::Mat& frame, glare_position& gp, glare_detector& gd) {
    // Photometric / Geometric / Priority 맵 생성
    cv::Mat gphoto = gd.computePhotometricMap(frame);
    cv::Mat ggeo   = gd.computeGeometricMap(gphoto);
    cv::Mat priority = gd.computePriorityMap(gphoto, ggeo);

    // 특정 priority에 해당하는 최댓값 컨투어의 바운딩 박스 반환
    auto extractBox = [&](int target_priority) -> cv::Rect2f {
        // 해당 priority만 마스크로 추출
        cv::Mat mask = (priority == target_priority);

        // 컨투어 탐색을 위한 이진 이미지 생성 (0/1 → 0/255)
        cv::Mat bin;
        mask.convertTo(bin, CV_8U, 255);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) return cv::Rect2f();

        // 가장 큰 컨투어 선택
        std::vector<cv::Point> maxContour;
        double maxArea = 0.0;
        for (const auto& c : contours) {
            double area = cv::contourArea(c);
            if (area > maxArea) {
                maxArea = area;
                maxContour = c;
            }
        }
        return cv::boundingRect(maxContour);
    };

    // Priority 1 먼저 확인 (영역 크기 제한: 너무 작거나 큰 영역 필터링)
    cv::Rect2f box1 = extractBox(1);
    if (box1.area() > 500.0f && box1.area() < 10000.0f)
        return { box1, 1 };

    // Priority 2 확인
    cv::Rect2f box2 = extractBox(2);
    if (box2.area() > 500.0f && box2.area() < 10000.0f)
        return { box2, 2 };

    // 감지 실패
    return { cv::Rect2f(), 0 };
}

int main() {
    namespace fs = std::filesystem;

    // 현재 파일(src/glare_evaluate.cpp) 기준 상대 경로
    const fs::path src_dir = fs::path(__FILE__).parent_path();
    const std::string img_dir   = (src_dir / "test_images").string();
    const std::string label_dir = (src_dir / "gt_labels").string();

    glare_detector gd;
    glare_position gp;

    // 성능 집계 변수
    float total_iou = 0.0f;
    int total_tp_iou = 0;   // IoU 평균을 TP에 한해 계산하기 위한 카운트
    int total_images = 0;
    int TP = 0, FP = 0, FN = 0, TN = 0;

    // mAP 계산(여기서는 단순 Precision@IoU-threshold 평균)용 카운터
    std::map<float, int> ap_tp;
    std::map<float, int> ap_fp;
    std::vector<float> iou_thresholds;
    for (float t = 0.5f; t <= 0.96f; t += 0.05f) iou_thresholds.push_back(t);

    // 시각화 결과 저장 (파일명, 이미지)
    std::vector<std::pair<std::string, cv::Mat>> results;
    std::vector<std::string> image_names;

    // === 이미지 루프 ===
    for (const auto& entry : fs::directory_iterator(img_dir)) {
        std::string img_path = entry.path().string();
        std::string fname = entry.path().stem().string();
        image_names.push_back(fname);

        std::string label_path = label_dir + "/" + fname + ".txt";

        // 원본 이미지 로드
        cv::Mat orig = cv::imread(img_path);
        if (orig.empty()) continue;

        // 시각화/연산용 리사이즈 프레임 생성
        cv::Mat frame;
        cv::resize(orig, frame, cv::Size(RESIZED_WIDTH, RESIZED_HEIGHT));

        // GT 라벨 로드 (원본 크기 기준) → 리사이즈 크기(RESIZED_WIDTH x RESIZED_HEIGHT)로 스케일링
        auto gt_boxes = loadYoloLabels(label_path, orig.cols, orig.rows);

        std::vector<cv::Rect2f> resized_gt_boxes;
        float x_scale = static_cast<float>(RESIZED_WIDTH) / orig.cols;
        float y_scale = static_cast<float>(RESIZED_HEIGHT) / orig.rows;
        for (const auto& box : gt_boxes) {
            resized_gt_boxes.emplace_back(
                box.x * x_scale,
                box.y * y_scale,
                box.width * x_scale,
                box.height * y_scale
            );
        }

        // 예측 박스 추출 (Priority 1 → 2 순서로 검사)
        auto [pred_box, priority_type] = getPredictedBox(frame, gp, gd);

        bool gt_has_glare   = !resized_gt_boxes.empty();
        bool pred_has_glare = (pred_box.area() > 1.0f);

        // === 정량 평가 (TP/FP/FN/TN, IoU, mAP 카운터) ===
        if (gt_has_glare && pred_has_glare) {
            // GT 여러 개 중 예측과의 최대 IoU 사용
            float best_iou = 0.0f;
            for (const auto& gt : resized_gt_boxes) {
                best_iou = std::max(best_iou, computeIoU(pred_box, gt));
            }

            // TP로 판정하는 IoU 기준(0.5)
            if (best_iou >= 0.5f) {
                TP++;
                total_tp_iou++;
                total_iou += best_iou;
            } else {
                FP++; // 예측은 했지만 GT와 충분히 겹치지 못함
            }

            // mAP용 임계값별 TP/FP 카운트
            for (float t : iou_thresholds) {
                if (best_iou >= t) ap_tp[t]++;
                else ap_fp[t]++;
            }
        } else if (!gt_has_glare && pred_has_glare) {
            // GT 없음 + 예측 있음 → FP
            FP++;
            for (float t : iou_thresholds) ap_fp[t]++;
        } else if (gt_has_glare && !pred_has_glare) {
            // GT 있음 + 예측 없음 → FN
            FN++;
        } else {
            // GT 없음 + 예측 없음 → TN
            TN++;
        }

        total_images++;

        // === 시각화 (GT: 빨강, Prediction: 파랑) ===
        for (const auto& gt : resized_gt_boxes) {
            cv::rectangle(frame, gt, cv::Scalar(0, 0, 255), 2); // Red
        }
        if (pred_has_glare) {
            // Priority 1/2 구분 없이 파란색 박스
            cv::rectangle(frame, pred_box, cv::Scalar(255, 0, 0), 2); // Blue
        }

        // 상단 상태 텍스트 (GT/Pred 존재 여부)
        std::string text = "GT: " + std::string(gt_has_glare ? "Yes" : "No") +
                           ", Pred: " + std::string(pred_has_glare ? "Yes" : "No");
        cv::putText(frame, text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 165, 255), 2);

        // === Legend (범례) : GT/Prediction(Glare)만 표시 ===
        int legend_x = 20, legend_y = 60; // 시작 좌표 (좌상단)
        int box_size = 20;                // 색상 박스 크기
        int spacing  = 30;                // 항목 간 세로 간격

        // Ground Truth (빨강)
        cv::rectangle(frame,
                      cv::Rect(legend_x, legend_y, box_size, box_size),
                      cv::Scalar(0, 0, 255), cv::FILLED);
        cv::putText(frame, "Ground Truth",
                    cv::Point(legend_x + box_size + 10, legend_y + box_size - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

        // Prediction (Glare, 파랑) — Priority 1/2 통합
        cv::rectangle(frame,
                      cv::Rect(legend_x, legend_y + spacing, box_size, box_size),
                      cv::Scalar(255, 0, 0), cv::FILLED);
        cv::putText(frame, "Prediction (Glare)",
                    cv::Point(legend_x + box_size + 10, legend_y + spacing + box_size - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);

        // 시각화 결과 저장
        results.emplace_back(fname, frame);
    }

    // === 최종 성능 출력 (Classification) ===
    std::cout << "\n=== Classification Evaluation ===" << std::endl;
    std::cout << "TP: " << TP << ", FP: " << FP << ", FN: " << FN << ", TN: " << TN << std::endl;

    int total = TP + FP + FN + TN;
    float precision = (TP + FP > 0) ? static_cast<float>(TP) / (TP + FP) : 0.0f;
    float recall    = (TP + FN > 0) ? static_cast<float>(TP) / (TP + FN) : 0.0f;
    float accuracy  = (total     > 0) ? static_cast<float>(TP + TN) / total : 0.0f;
    float f1 = (precision + recall > 0.0f) ? 2.0f * precision * recall / (precision + recall) : 0.0f;

    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Recall: "    << recall    << std::endl;
    std::cout << "F1 Score: "  << f1        << std::endl;
    std::cout << "Accuracy: "  << accuracy  << std::endl;

    // === IoU 평균 (TP에 한해) ===
    std::cout << "\n=== IoU Evaluation (Glare Exists) ===" << std::endl;
    std::cout << "Total Images with GT: " << total_tp_iou << std::endl;
    std::cout << "Average IoU (on TP): "
              << (total_tp_iou > 0 ? total_iou / total_tp_iou : 0.0f) << std::endl;

    // === mAP 평가 (간이 Precision 평균) ===
    std::cout << "\n=== mAP Evaluation ===" << std::endl;
    float ap_sum = 0.0f;
    int ap_count = 0;
    for (float t : iou_thresholds) {
        int tp = ap_tp[t];
        int fp = ap_fp[t];
        float ap_precision = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
        std::cout << "IoU >= " << t << " : Precision = " << ap_precision << std::endl;
        ap_sum += ap_precision;
        ap_count++;
    }
    float mAP = (ap_count > 0) ? ap_sum / ap_count : 0.0f;
    std::cout << "mAP@[0.5:0.95] = " << mAP << std::endl;

    // === 이미지 결과 탐색 UI (좌/우 화살표, ESC 종료) ===
    int idx = 0;
    while (!results.empty()) {
        const auto& [name, img] = results[idx];
        cv::imshow("Result", img);
        std::cout << "Viewing: " << name
                  << " - Press LEFT/RIGHT to navigate, ESC to exit." << std::endl;

        int key = cv::waitKey(0);
        if (key == 27) break; // ESC
        else if (key == 81 || key == 2424832) { // Left arrow (Linux/Win)
            idx = (idx - 1 + static_cast<int>(results.size())) % static_cast<int>(results.size());
        } else if (key == 83 || key == 2555904) { // Right arrow (Linux/Win)
            idx = (idx + 1) % static_cast<int>(results.size());
        }
    }

    return 0;
}
