#pragma execution_character_set("utf-8")

#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
    // ---------------- 1. 读图 ----------------
    const string imagePath = "hole.jpg";
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        cout << "图片加载失败！找不到 " << imagePath << endl;
        cout << "当前工作目录: " << std::filesystem::current_path().string() << endl;
        return -1;
    }

    if (img.cols > 1000) {
        double scale = 1000.0 / img.cols;
        cv::resize(img, img, cv::Size(), scale, scale);
    }
    cout << "图片尺寸: " << img.cols << " x " << img.rows << endl;

    // ---------------- 2. 灰度 + 二值化 ----------------
    // 孔是暗的、盘面是亮的，所以用 THRESH_BINARY_INV 把孔变成白色前景。
    // 注意：这里孔是被当成"暗色物体"检测的，不是去挖轮廓内部的空洞，
    // 所以仍然用 RETR_EXTERNAL 就够，不需要 hierarchy。
    cv::Mat gray, blurImg, binary;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurImg, cv::Size(5, 5), 0);
    cv::threshold(blurImg, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 开运算去噪、闭运算把孔边缘的小毛刺补圆
    cv::Mat k3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, k3);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, k3);

    // ---------------- 3. 找轮廓 ----------------
    vector<vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cout << "外轮廓总数: " << contours.size() << endl;

    // ---------------- 4. 三重筛选，只留"完整的圆孔" ----------------
    // 本图有三个坑，必须同时挡住：
    //   ① 图片四周有一圈黑色边框，它和左侧暗区、中央大孔连成一块约占全图 40% 的
    //      连通域，还贴着图像边缘 —— 用"碰到边缘就丢弃"挡掉；
    //   ② 顶部和底部被画面切掉一半的孔 —— 同样靠"碰边"挡掉（它们不完整）；
    //   ③ 左侧两大块不规则的暗区、右侧中央大沉孔 —— 用面积上限 + 圆度挡掉。
    // 实测 4 个完整孔：面积 1816 ~ 5955（占全图 0.36% ~ 1.19%），圆度接近 1.0。
    double minArea = img.rows * img.cols * 0.0015; // 0.15%
    double maxArea = img.rows * img.cols * 0.03;   // 3%
    const int BORDER = 2;                          // 距边缘 2 像素内算"贴边"
    cout << "有效面积区间: " << minArea << " ~ " << maxArea << endl;

    cv::Mat result = img.clone();
    int count = 0;

    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect box = cv::boundingRect(contours[i]);
        double area = cv::contourArea(contours[i]);

        // ① ② 贴边的都不要：要么是外框，要么是被画面切断的半个孔
        bool touchBorder = box.x <= BORDER || box.y <= BORDER ||
                           box.x + box.width >= img.cols - BORDER ||
                           box.y + box.height >= img.rows - BORDER;
        if (touchBorder) {
            cout << "第 " << i + 1 << " 个轮廓面积 " << area << "  -> 忽略(贴边/不完整)" << endl;
            continue;
        }

        if (area <= minArea || area >= maxArea) {
            cout << "第 " << i + 1 << " 个轮廓面积 " << area << "  -> 忽略(尺寸不符)" << endl;
            continue;
        }

        // ③ 圆度：面积 / 最小外接圆面积，标准圆约等于 1.0
        cv::Point2f center;
        float radius = 0;
        cv::minEnclosingCircle(contours[i], center, radius);
        double roundness = area / (CV_PI * radius * radius);
        if (roundness < 0.75) {
            cout << "第 " << i + 1 << " 个轮廓面积 " << area << " 圆度 " << roundness
                 << "  -> 忽略(不是圆)" << endl;
            continue;
        }

        cout << "第 " << i + 1 << " 个轮廓面积 " << area << " 圆度 " << roundness
             << "  -> 计为孔" << endl;
        count++;

        cv::circle(result, center, (int)radius, cv::Scalar(0, 0, 255), 2);
        cv::circle(result, center, 3, cv::Scalar(0, 0, 255), -1);
        cv::putText(result, to_string(count),
            cv::Point((int)(center.x - 10), (int)(center.y + 8)),
            cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
    }

    cv::putText(result, "Holes: " + to_string(count), cv::Point(20, 50),
        cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 255), 3);

    cout << "======================================" << endl;
    cout << "识别完成，共找到 " << count << " 个完整的孔。" << endl;
    cout << "======================================" << endl;

    cv::imwrite("result_holes.png", result);

    cv::imshow("1. Binary Image", binary);
    cv::imshow("2. Holes Result", result);
    cv::waitKey(0);
    return 0;
}
