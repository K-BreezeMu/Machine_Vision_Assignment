#pragma execution_character_set("utf-8")

#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
    // ---------------- 1. 读图 ----------------
    const string imagePath = "paperclips.jpg";
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        cout << "图片加载失败！确认当前工作目录下有 " << imagePath << endl;
        return -1;
    }

    // 窗口太大屏幕放不下时等比缩小（后面的面积阈值按比例算，缩放不影响判断）
    if (img.cols > 1000) {
        double scale = 1000.0 / img.cols;
        cv::resize(img, img, cv::Size(), scale, scale);
    }
    cout << "图片尺寸: " << img.cols << " x " << img.rows << endl;

    // ---------------- 2. 灰度 + 二值化 ----------------
    // 回形针是深色、背景是浅色，用 THRESH_BINARY_INV 让回形针变成白色前景
    cv::Mat gray, binary;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 闭运算：补上管子上的小断口和反光孔，让每枚回形针连成一个整体
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

    // ---------------- 3. 找轮廓 ----------------
    vector<vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cout << "外轮廓总数: " << contours.size() << endl;

    // ---------------- 4. 面积过滤 ----------------
    // 这里是之前数出 0 的地方，原因是回形针"空心"：
    //   RETR_EXTERNAL 拿到的是最外层轮廓，contourArea 算的是这条闭合曲线
    //   "围住"的整片面积，管子围出的中空部分也算在内。
    // 本图实测：管壁像素面积只有 6000~7300，但外轮廓面积是 10514~11226，
    // 所以原来 (area > 500 && area < 10000) 的上界压在真实值下方，13 枚全被筛掉。
    // 改成按整图面积的比例来卡：回形针约占整图 2%，取 0.6% ~ 6% 留足余量。
    double minArea = img.rows * img.cols * 0.006; // 小于 0.6%：反光碎屑、噪点
    double maxArea = img.rows * img.cols * 0.06;  // 大于 6%  ：背景大块、多枚粘连
    cout << "有效面积区间: " << minArea << " ~ " << maxArea << endl;

    // ---------------- 5. 画框 + 编号 ----------------
    cv::Mat result = img.clone();
    int count = 0;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);

        if (area <= minArea || area >= maxArea) {
            cout << "第 " << i + 1 << " 个轮廓面积 " << area << "  -> 忽略" << endl;
            continue;
        }

        cout << "第 " << i + 1 << " 个轮廓面积 " << area << "  -> 计为回形针" << endl;
        count++;

        // 最小外接矩形比正矩形更贴合斜放的回形针
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        cv::Point2f pts[4];
        rect.points(pts);
        for (int j = 0; j < 4; j++) {
            cv::line(result, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 0, 255), 2);
        }

        cv::Point2f c = rect.center;
        cv::putText(result, to_string(count),
            cv::Point((int)(c.x - 10), (int)(c.y + 8)),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
    }

    cv::putText(result, "Total: " + to_string(count), cv::Point(20, 50),
        cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 255), 3);

    cout << "======================================" << endl;
    cout << "识别完成，共找到 " << count << " 个回形针。" << endl;
    cout << "======================================" << endl;

    cv::imwrite("result.png", result); // 窗口关掉也能回看结果

    cv::imshow("1. Binary Image", binary);
    cv::imshow("2. Final Result", result);
    cv::waitKey(0);
    return 0;
}
