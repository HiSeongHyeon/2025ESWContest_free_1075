// clang-format off

#include "coordinates.hpp"
#include "config.hpp"
#include "get_grid_size.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>

inline double to_radians(double degrees) { return degrees * CV_PI / 180.0; }

cv::Point3d ray_plane_intersection(const cv::Point3d &origin,
                                   const cv::Vec3d &direction, double plane_z) {
  // Ray 방정식: P = origin + t * direction
  // 평면 방정식: P.z = plane_z
  // origin.z + t * direction[2] = plane_z
  // t = (plane_z - origin.z) / direction[2]
  if (std::abs(direction[2]) <
      1e-9) { // z 방향 성분이 거의 0이면 평면과 평행 (또는 포함)
    return cv::Point3d(std::numeric_limits<double>::max(),
                       std::numeric_limits<double>::max(),
                       std::numeric_limits<double>::max());
  }
  double t = (plane_z - origin.z) /
             direction[2]; // origin.y -> origin.z, direction[1] -> direction[2]
  return origin + t * cv::Point3d(direction);
}

std::vector<int> to_bit_list(const std::pair<int, int> &grid_coord, int bits) {
  // grid_coord.first는 col (x 좌표), grid_coord.second는 row (y 좌표)
  int col = grid_coord.first;
  int row = grid_coord.second;

  if (col < 0 || col > 2 || row < 0 || row > 2) {
    // Handle error: invalid input for 3x3 grid
    return {0, 0, 0, 0}; // Example error return
  }

  std::vector<int> bit_list;
  bit_list.reserve(4); // 항상 4개의 비트

  // 상위 2비트: Column (x 좌표) [Col_MSB, Col_LSB]
  bit_list.push_back((col >> 1) & 1); // Col_MSB (col의 1번째 비트)
  bit_list.push_back(col & 1);        // Col_LSB (col의 0번째 비트)

  // 하위 2비트: Row (y 좌표) [Row_MSB, Row_LSB]
  bit_list.push_back((row >> 1) & 1); // Row_MSB (row의 1번째 비트)
  bit_list.push_back(row & 1);        // Row_LSB (row의 0번째 비트)

  return bit_list;
}

std::pair<int, int> camera_to_driver_coords(
    const std::pair<double, double> &sun_center,
    const std::pair<int, int> &image_size, const std::pair<double, double> &fov,
    const cv::Point3d &camera_pos, const cv::Point3d &driver_eye_pos,
    double windshield_z, const std::pair<double, double> &glass_size,
    const std::pair<double, double> &glass_origin) {

  cv::Mat sun_center_mat(1, 1, CV_64FC2);
  sun_center_mat.at<cv::Vec2d>(0, 0)[0] = sun_center.first;
  sun_center_mat.at<cv::Vec2d>(0, 0)[1] = sun_center.second;

  cv::Mat undistorted_points;
  cv::undistortPoints(sun_center_mat, undistorted_points, DEFAULT_CAMERA_MATRIX,
                      DEFAULT_DISTORTION_COEFFICIENTS, cv::noArray(),
                      DEFAULT_CAMERA_MATRIX);

  cv::Vec2d corrected_sun_vec = undistorted_points.at<cv::Vec2d>(0, 0);
  std::pair<double, double> corrected_sun_center = {corrected_sun_vec[0],
                                                    corrected_sun_vec[1]};

  double img_w = static_cast<double>(image_size.first);
  double img_h = static_cast<double>(image_size.second);
  double fov_x_deg = fov.first;
  double fov_y_deg = fov.second;
  double x_center_px = sun_center.first;
  double y_center_px = sun_center.second;
  // double x_center_px = corrected_sun_center.first;
  // double y_center_px = corrected_sun_center.second;

  double norm_x = (x_center_px / img_w - 0.5) * 2.0;
  double norm_y = (y_center_px / img_h - 0.5) * 2.0;

  double angle_x_rad = to_radians(norm_x * (fov_x_deg / 2.0));
  double angle_y_rad = to_radians(norm_y * (fov_y_deg / 2.0));

  // (du,dv,dw)는 방향 벡터, 좌표계 축은 (x,y,z)와 동일
  double du = std::tan(angle_x_rad);
  double dv = -std::tan(angle_y_rad);
  double dw = 1.0; // 카메라 기본 전방은 +z(+w) 방향
  
  cv::Vec3d D_no_rotation(du, dv, dw);

  double pitch_rad = to_radians(DEFAULT_CAMERA_PITCH_DEGREES);

  // X축 기준 회전 행렬 생성
  // Rx = [1  0       0     ]
  //      [0  cos(p) -sin(p)]
  //      [0  sin(p)  cos(p)]
  cv::Matx33d rotation_matrix_x(
        1, 0, 0,
        0, std::cos(pitch_rad), -std::sin(pitch_rad),
        0, std::sin(pitch_rad), std::cos(pitch_rad)
  );

  // 기본 방향 벡터에 회전 행렬 적용
  cv::Vec3d D_rotated = rotation_matrix_x * D_no_rotation;
  cv::Vec3d D_vec = cv::normalize(D_rotated); // 최종 방향 벡터 정규화

  // 광선의 원점은 driver_eye_pos로 설정
  cv::Point3d intersection =
      ray_plane_intersection(driver_eye_pos, D_vec, windshield_z);
  double px = intersection.x;
  double py = intersection.y; // 이전에는 _ 로 무시, 이제 py 사용
  double pz = intersection.z; // windshield_z 와 동일

  // 물리적 교차점 (px, py)을 glass_origin과 glass_size를 기반으로 grid 좌표로
  // 매핑
  std::pair<int, int> grid_dims = get_grid_size();
  int grid_cols = grid_dims.first;
  int grid_rows = grid_dims.second;

  double glass_width = glass_size.first;
  double glass_height = glass_size.second;
  double x_left_glass =
      glass_origin.first; // glass_origin은 좌상단 기준 (x_left, y_top)
  double y_top_glass = glass_origin.second;

  int grid_x =
      static_cast<int>(((px - x_left_glass) / glass_width) * grid_cols);
  int grid_y =
      static_cast<int>(((y_top_glass - py) / glass_height) * grid_rows);

  grid_x = std::max(0, std::min(grid_x, grid_cols - 1));
  grid_y = std::max(0, std::min(grid_y, grid_rows - 1));

  return {grid_x, grid_y};
}

// void visualize_grid_on_frame(const cv::Mat &display_frame_in,
//                              cv::Mat &display_frame_out,
//                              const std::pair<int, int> &total_grid_dims,
//                              int step_size) {
//   if (display_frame_in.empty() || step_size <= 0) {
//     if (!display_frame_in.empty())
//       display_frame_in.copyTo(display_frame_out);
//     return;
//   }

//   display_frame_in.copyTo(
//       display_frame_out); // 원본에 그리지 않도록 복사본 사용 또는 직접 그림

//   int img_width = display_frame_in.cols;
//   int img_height = display_frame_in.rows;
//   int total_cols = total_grid_dims.first;
//   int total_rows = total_grid_dims.second;

//   // 각 그리드 인덱스에 대한 색상을 미리 정의
//   std::vector<cv::Scalar> grid_colors;
//   for (int r = 0; r < total_rows; ++r) {
//     for (int c = 0; c < total_cols; ++c) {
//       // 간단한 색상 생성 로직
//       grid_colors.push_back(
//           cv::Scalar((c * 255 / total_cols), (r * 255 / total_rows), 200));
//     }
//   }
//   if (grid_colors.empty())
//     return; // 그리드가 없으면 종료

//   // 이미지의 각 영역(step_size 간격)을 순회
//   for (int y_pixel = 0; y_pixel < img_height; y_pixel += step_size) {
//     for (int x_pixel = 0; x_pixel < img_width; x_pixel += step_size) {
//       // 현재 픽셀 블록의 중심 좌표
//       double current_pixel_u = static_cast<double>(x_pixel + step_size / 2);
//       double current_pixel_v = static_cast<double>(y_pixel + step_size / 2);

//       if (current_pixel_u >= img_width || current_pixel_v >= img_height)
//         continue;

//       // 이 픽셀 좌표가 어떤 그리드 셀로 매핑되는지 계산
//       std::pair<double, double> sun_center_for_transform = {current_pixel_u,
//                                                             current_pixel_v};
//       std::pair<int, int> mapped_grid =
//           camera_to_driver_coords(sun_center_for_transform);

//       if (mapped_grid.first != -1 &&
//           mapped_grid.second != -1) { // 유효한 그리드 좌표로 매핑된 경우
//         // 1차원 그리드 인덱스로 변환
//         int grid_1d_idx = mapped_grid.second * total_cols + mapped_grid.first;

//         if (grid_1d_idx >= 0 && grid_1d_idx < grid_colors.size()) {
//           // 해당 픽셀 블록을 매핑된 그리드의 색상으로 칠하기
//           cv::Rect roi(x_pixel, y_pixel, step_size, step_size);

//           // 반투명하게 칠하기
//           cv::Mat roi_mat = display_frame_out(roi);
//           cv::Mat color_roi(roi.size(), CV_8UC3, grid_colors[grid_1d_idx]);
//           double alpha = 0.4;
//           cv::addWeighted(color_roi, alpha, roi_mat, 1.0 - alpha, 0.0, roi_mat);
//         }
//       }
//     }
//   }
// }

void visualize_grid_on_frame(const cv::Mat &display_frame_in,
                             cv::Mat &display_frame_out,
                             const std::pair<int, int> &total_grid_dims,
                             int step_size) { // step_size는 호환성을 위해
    
    if (display_frame_in.empty()) {
        return;
    }

    // 1. 초기 설정 및 데이터 구조 준비
    display_frame_in.copyTo(display_frame_out); // 원본 이미지를 출력 Mat에 복사
    int img_w = display_frame_in.cols;
    int img_h = display_frame_in.rows;
    int total_cols = total_grid_dims.first;
    int total_rows = total_grid_dims.second;

    std::vector<std::vector<cv::Point>> points_per_grid(total_cols * total_rows);
    cv::Mat grid_coords_map(img_h, img_w, CV_32SC2);

    std::vector<std::vector<int>> vertical_boundary_x_coords(total_cols - 1);
    std::vector<std::vector<cv::Point>> horizontal_boundary_points(total_rows - 1);

    // 2. 모든 픽셀에 대한 그리드 좌표 계산
    for (int y = 0; y < img_h; ++y) {
        for (int x = 0; x < img_w; ++x) {
            std::pair<int, int> grid_coord = camera_to_driver_coords(
                {static_cast<double>(x), static_cast<double>(y)}
            );
            
            int gx = grid_coord.first;
            int gy = grid_coord.second;
            
            grid_coords_map.at<cv::Vec2i>(y, x) = cv::Vec2i(gx, gy);
            if (gy * total_cols + gx < points_per_grid.size()) {
                points_per_grid[gy * total_cols + gx].push_back(cv::Point(x, y));
            }
        }
    }

    // 3. 그리드 맵을 순회하며 경계점 찾기
    for (int y = 0; y < img_h - 1; ++y) {
        for (int x = 0; x < img_w - 1; ++x) {
            cv::Vec2i current_grid = grid_coords_map.at<cv::Vec2i>(y, x);
            int gx_curr = current_grid[0];
            int gy_curr = current_grid[1];

            int gx_next = grid_coords_map.at<cv::Vec2i>(y, x + 1)[0];
            if (gx_curr != gx_next) {
                int boundary_idx = std::min(gx_curr, gx_next);
                if (boundary_idx < vertical_boundary_x_coords.size()) {
                    vertical_boundary_x_coords[boundary_idx].push_back(x + 1);
                }
            }

            int gy_next = grid_coords_map.at<cv::Vec2i>(y + 1, x)[1];
            if (gy_curr != gy_next) {
                int boundary_idx = std::min(gy_curr, gy_next);
                if (boundary_idx < horizontal_boundary_points.size()) {
                    horizontal_boundary_points[boundary_idx].push_back(cv::Point(x, y + 1));
                }
            }
        }
    }

    // 4. 경계선 및 숫자 그리기
    cv::Scalar line_color(255, 255, 255); // 흰색
    int line_thickness = 3; // 두께 조절

    // 수직선 (평균 x 좌표를 계산하여 직선으로 그림)
    std::vector<int> avg_vertical_x; // 숫자 정렬에 재사용하기 위해 별도 저장
    for (const auto& x_coords : vertical_boundary_x_coords) {
        if (!x_coords.empty()) {
            long long sum = std::accumulate(x_coords.begin(), x_coords.end(), 0LL);
            int avg_x = static_cast<int>(sum / x_coords.size());
            avg_vertical_x.push_back(avg_x); // 평균값 저장
            cv::line(display_frame_out, cv::Point(avg_x, 0), cv::Point(avg_x, img_h - 1), line_color, line_thickness, cv::LINE_AA);
        }
    }

    // 수평선 (찾은 모든 경계점을 그대로 그려 자연스러운 곡선)
    for (const auto& points : horizontal_boundary_points) {
        for (const auto& point : points) {
            // 점을 찍어 선처럼 보이게 함
            cv::circle(display_frame_out, point, 1, line_color, -1, cv::LINE_AA);
        }
    }

    // 열(column) 경계 정의
    std::vector<int> col_bounds = {0};
    col_bounds.insert(col_bounds.end(), avg_vertical_x.begin(), avg_vertical_x.end());
    col_bounds.push_back(img_w);

    // 행(row) 경계 정의 (각 수평선의 전체 평균 y 사용)
    std::vector<int> avg_horizontal_y_overall;
    for (const auto& points : horizontal_boundary_points) {
        if (!points.empty()) {
            long long sum_y = 0;
            for (const auto& p : points) {
                sum_y += p.y;
            }
            avg_horizontal_y_overall.push_back(static_cast<int>(sum_y / points.size()));
        }
    }
    std::vector<int> row_bounds = {0};
    row_bounds.insert(row_bounds.end(), avg_horizontal_y_overall.begin(), avg_horizontal_y_overall.end());
    row_bounds.push_back(img_h);
    
    // 그리드 번호 그리기
    for (int i = 0; i < total_cols * total_rows; ++i) {
        int c = i % total_cols; // 현재 셀의 열(column) 인덱스
        int r = i / total_cols; // 현재 셀의 행(row) 인덱스

        // 안전 장치: 경계 배열의 크기를 초과하지 않도록 확인
        if (c + 1 >= col_bounds.size() || r + 1 >= row_bounds.size()) continue;

        // 이상적인 그리드 셀의 우측 하단 좌표
        int ideal_right = col_bounds[c + 1];
        int ideal_bottom = row_bounds[r + 1];

        std::string number_text = std::to_string(i + 1);
        double font_scale = 1.0;     // 크기
        int font_thickness = 2;      // 두께 조절
        int padding = 10;            // 여백
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(number_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
        
        // 이상적인 경계를 기준으로 텍스트 위치 계산
        cv::Point text_origin(
            ideal_right - text_size.width - padding,
            ideal_bottom - padding
        );
        
        cv::Scalar text_color(0, 0, 255); // BGR
        
        // 검은색 외곽선(가독성) -> 빨간색 텍스트
        cv::putText(display_frame_out, number_text, text_origin, cv::FONT_HERSHEY_SIMPLEX, font_scale, 
                    cv::Scalar(0, 0, 0), font_thickness + 2, cv::LINE_AA);
        cv::putText(display_frame_out, number_text, text_origin, cv::FONT_HERSHEY_SIMPLEX, font_scale, 
                    text_color, font_thickness, cv::LINE_AA);
    }
}