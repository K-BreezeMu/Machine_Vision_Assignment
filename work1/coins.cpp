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
    const string imagePath = "coins.png";
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        cout << "图片加载失败！找不到 " << imagePath << endl;
        cout << "当前工作目录: " << std::filesystem::current_path().string() << endl;
        cout << "请把硬币图存成 coins.png 放到上面这个目录（一般是 Project1）" << endl;
        return -1;
    }

    if (img.cols > 1000) {
        double scale = 1000.0 / img.cols;
        cv::resize(img, img, cv::Size(), scale, scale);
    }
    cout << "图片尺寸: " << img.cols << " x " << img.rows << endl;

    // ---------------- 2. 灰度 + 二值化 ----------------
    // 和回形针那张图相反：这里是"亮硬币 + 暗桌面"，
    // 所以要用 THRESH_BINARY（亮的变白、暗的变黑），不能用 THRESH_BINARY_INV。
    // 先做 5x5 高斯模糊，压掉桌面上的颗粒噪点，Otsu 的阈值会更稳。
    cv::Mat gray, blurImg, binary;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurImg, cv::Size(5, 5), 0);
    cv::threshold(blurImg, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 开运算去掉零星噪点，闭运算把硬币边缘的小缺口补圆
    cv::Mat k3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat k5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, k3);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, k5);

    // ---------------- 3. 找轮廓 ----------------
    vector<vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cout << "外轮廓总数: " << contours.size() << endl;

    // ---------------- 4. 面积过滤 ----------------
    // 硬币是实心圆盘，contourArea 约等于它的实际像素面积（不像回形针有空心部分）。
    // 本图实测：10 枚硬币面积 6119 ~ 15209，占整图 1.6% ~ 4.0%，
    // 其中顶部中间那枚颜色较暗，二值化时被咬掉一块，所以只有 6119。
    // 因此下界放到 0.8%，给偏暗的硬币留余量；上界 8% 用来挡掉粘连和背景。
    double minArea = img.rows * img.cols * 0.008;
    double maxArea = img.rows * img.cols * 0.08;
    cout << "有效面积区间: " << minArea << " ~ " << maxArea << endl;

    // ---------------- 5. 画圆 + 编号 ----------------
    cv::Mat result = img.clone();
    int count = 0;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);

        if (area <= minArea || area >= maxArea) {
            cout << "第 " << i + 1 << " 个轮廓面积 " << area << "  -> 忽略" << endl;
            continue;
        }

        cout << "第 " << i + 1 << " 个轮廓面积 " << area << "  -> 计为硬币" << endl;
        count++;

        // 硬币是圆的，用最小外接圆比矩形框更贴切
        cv::Point2f center;
        float radius = 0;
        cv::minEnclosingCircle(contours[i], center, radius);
        cv::circle(result, center, (int)radius, cv::Scalar(0, 0, 255), 2);
        cv::circle(result, center, 3, cv::Scalar(0, 0, 255), -1);

        cv::putText(result, to_string(count),
            cv::Point((int)(center.x - 10), (int)(center.y + 8)),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }

    cv::putText(result, "Total: " + to_string(count), cv::Point(20, 50),
        cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 255), 3);

    cout << "======================================" << endl;
    cout << "识别完成，共找到 " << count << " 枚硬币。" << endl;
    cout << "======================================" << endl;

    cv::imwrite("result_coins.png", result);

    cv::imshow("1. Binary Image", binary);
    cv::imshow("2. Coins Result", result);
    cv::waitKey(0);
    return 0;
}
