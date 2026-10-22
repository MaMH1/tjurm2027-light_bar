// ============================================================================
// 任务二：装甲板灯条识别
// 流水线（对应 PDF 要求）：
//   1. 设置图像掩码二值化：HSV 提取红色（灯条是红色 LED）
//   2. cv::findContours 找轮廓
//   3. cv::minAreaRect 拟合每个灯条，算出上下两个端点坐标
//   4. 把两个灯条的四个端点按顺序连线，画出装甲板矩形
// 运行方式（在项目根目录 ~/light_bar 下）：
//   ./build/light_bar       处理整个视频 -> output.mp4 + out_frames/ 逐帧结果图
//   ./build/light_bar show  额外用 cv::imshow 弹窗实时显示（按 ESC 退出）
// ============================================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

// 一个灯条：最小外接矩形 + 上下两个端点
struct LightBar {
    cv::RotatedRect rect;   // cv::minAreaRect 拟合出的外接矩形
    cv::Point2f top;        // 上端点（灯条上边缘的中点）
    cv::Point2f bottom;     // 下端点（灯条下边缘的中点）
};

// 从一帧图像中检测红色灯条，结果按 x 坐标从小到大排序（左边灯条在前）
// rTh / dTh：颜色阈值（R 通道 > rTh 且 R-B > dTh）。
// 默认 (25, 8) 偏松：前 30 帧灯条较暗也能检到；带进的噪声由 findBestPair 过滤。
void detectLightBars(const cv::Mat& frame, std::vector<LightBar>& bars,
                     double rTh = 25.0, double dTh = 8.0) {
    bars.clear();

    // ---- 第一步：图像掩码 + 二值化 ----
    // 灯条是红色 LED：R 通道亮、B 通道暗。用通道差 R-B 提取红色，
    // 对亮度变化比 HSV 更稳定（实测 HSV 在暗帧漏检严重）。
    std::vector<cv::Mat> ch;
    cv::split(frame, ch);  // ch[0]=B ch[1]=G ch[2]=R
    cv::Mat diff;
    cv::subtract(ch[2], ch[0], diff);   // diff = R - B（减到 0 为止）
    cv::Mat mask = (ch[2] > rTh) & (diff > dTh);

    // 形态学处理：先开运算去掉孤立噪点，再闭运算填掉灯条内部的小空洞
    cv::Mat k3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat k5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k3);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k5);

    // ---- 第二步：cv::findContours 找轮廓 ----
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // ---- 第三步：cv::minAreaRect 拟合灯条并计算上下端点 ----
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < 50 || area > 5000) continue;   // 面积太小是噪声，太大不是灯条

        cv::RotatedRect rr = cv::minAreaRect(c);
        float w = std::min(rr.size.width, rr.size.height);   // 短边（灯条宽度）
        float h = std::max(rr.size.width, rr.size.height);   // 长边（灯条高度）
        // 不单独限制灯条高度：面积过滤 + 长宽比已覆盖（h<10 时面积必 <50，
        // h>300 时面积必 >5000），单独限高反而会误杀远处变小了的灯条
        if (h / w < 2.2) continue;                           // 灯条必须是细长的

        // minAreaRect 给出 4 个顶点，按 y 排序：
        // 上面两个点的平均值 = 上端点，下面两个点的平均值 = 下端点
        cv::Point2f pts[4];
        rr.points(pts);
        std::sort(pts, pts + 4,
                  [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

        LightBar bar;
        bar.rect = rr;
        bar.top = (pts[0] + pts[1]) * 0.5f;
        bar.bottom = (pts[2] + pts[3]) * 0.5f;
        bars.push_back(bar);
    }

    // 按 x 排序：bars[0] 是左灯条，bars[1] 是右灯条
    std::sort(bars.begin(), bars.end(),
              [](const LightBar& a, const LightBar& b) {
                  return a.rect.center.x < b.rect.center.x;
              });
}

// 从候选灯条中挑"最像同一块装甲板的一对"：一左一右、高度接近、上下对齐、
// 水平间距不能太近（太近说明是同一个灯条的碎片）
bool findBestPair(const std::vector<LightBar>& bars, int& iL, int& iR) {
    iL = iR = -1;
    double bestScore = -1;
    for (size_t i = 0; i < bars.size(); ++i) {
        float hi = std::max(bars[i].rect.size.width, bars[i].rect.size.height);
        for (size_t j = i + 1; j < bars.size(); ++j) {
            float hj = std::max(bars[j].rect.size.width, bars[j].rect.size.height);
            if (std::min(hi, hj) / std::max(hi, hj) < 0.5f) continue;  // 高度差太多
            float dy = std::fabs(bars[i].rect.center.y - bars[j].rect.center.y);
            if (dy > 0.5f * std::min(hi, hj)) continue;                // 上下错开太多
            float dx = std::fabs(bars[i].rect.center.x - bars[j].rect.center.x);
            if (dx < 0.5f * std::min(hi, hj)) continue;                // 靠太近（同一灯条碎片）
            // 评分：灯条越大、上下越对齐得分越高
            double score = std::min(hi, hj) * (1.0 - dy / std::min(hi, hj));
            if (score > bestScore) { bestScore = score; iL = (int)i; iR = (int)j; }
        }
    }
    return iL >= 0;
}

// ---- 第四步：四个端点按顺序连线画矩形 ----
// 顺序：左上 -> 左下 -> 右下 -> 右上 -> 回到左上
void drawArmor(cv::Mat& img, const LightBar& left, const LightBar& right) {
    cv::Point corners[4] = {left.top, left.bottom, right.bottom, right.top};
    for (int i = 0; i < 4; ++i)
        cv::line(img, corners[i], corners[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);

    // 把四个端点用小圆点标出来，方便观察
    for (int i = 0; i < 4; ++i)
        cv::circle(img, corners[i], 4, cv::Scalar(0, 0, 255), -1);
}

// ============================================================================
// 选做部分：PnP 位姿解算 + 绘制三维坐标轴
// ============================================================================

// 相机内参（PDF 给出）：fx, cx, fy, cy
const double CAM_K[9] = {9.28130989e+02, 0,              3.77572945e+02,
                         0,              9.30138391e+02, 2.83892859e+02,
                         0,              0,              1.0000};
// 畸变系数（PDF 给出）：k1, k2, p1, p2, k3
const double CAM_DIST[5] = {-2.54433647e-01, 5.69431382e-01, 3.65405229e-03,
                            -1.09433818e-03, -1.33846840e+00};

// 小装甲板模型（单位 mm）。FLU 轴序：x 指向装甲板前方，y 指向左侧，z 指向上方
const double HALF_W = 62.5;   // 两灯条中心距的一半（小装甲板 135mm 宽，灯条内缩 5mm）
const double HALF_L = 27.5;   // 灯条长度的一半（灯条长 55mm）

// 用四个端点做 PnP 位姿解算，成功时把旋转/平移向量填入 rvec/tvec
bool solveArmorPose(const LightBar& left, const LightBar& right,
                    const cv::Mat& K, const cv::Mat& dist,
                    cv::Mat& rvec, cv::Mat& tvec) {
    // 3D 点（装甲板坐标系）与 2D 点（图像）按顺序一一对应：
    // 左灯条上端点、左灯条下端点、右灯条下端点、右灯条上端点
    std::vector<cv::Point3f> obj = {
        {0.f, (float)HALF_W,  (float)HALF_L},    // 左灯条上端点（板左侧 = +y）
        {0.f, (float)HALF_W, -(float)HALF_L},    // 左灯条下端点
        {0.f, -(float)HALF_W, -(float)HALF_L},   // 右灯条下端点（板右侧 = -y）
        {0.f, -(float)HALF_W,  (float)HALF_L},   // 右灯条上端点
    };
    std::vector<cv::Point2f> img = {left.top, left.bottom, right.bottom, right.top};

    // IPPE 专门用于共面点的位姿解算，solvePnPGeneric 会给出全部候选解
    // （共面点有两个解）；装甲板在相机前方时 tvec 的 z 分量为正，据此选出正确的解
    std::vector<cv::Mat> rvecs, tvecs;
    int n = cv::solvePnPGeneric(obj, img, K, dist, rvecs, tvecs,
                                false, cv::SOLVEPNP_IPPE);
    int best = -1;
    for (int i = 0; i < n; ++i)
        if (tvecs[i].at<double>(2) > 0) { best = i; break; }

    if (best < 0) {
        // IPPE 没给出在相机前方的解时，退回通用的迭代法再试一次
        bool ok = cv::solvePnP(obj, img, K, dist, rvec, tvec, false,
                               cv::SOLVEPNP_ITERATIVE);
        if (!ok || tvec.at<double>(2) < 0) return false;
    } else {
        rvec = rvecs[best];
        tvec = tvecs[best];
    }

    // 共面点的 PnP 有两个解：装甲板正面朝向相机 / 背对相机，两者投影完全相同
    // （装甲板模型左右、上下对称）。视频里灯条发光面正对镜头，法向量应指向
    // 相机（z 分量为负）。若解出的法向量背离相机，把位姿绕装甲板自身 z 轴
    // 翻转 180 度：法向量反向、y 轴反向、z 轴不变，平移向量也不变。
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    if (R.at<double>(2, 0) > 0) {
        R = R * (cv::Mat_<double>(3, 3) << -1, 0, 0, 0, -1, 0, 0, 0, 1);
        cv::Rodrigues(R, rvec);
    }
    return true;
}

// 在装甲板中心画三维坐标轴（轴长 60mm）：x 前=红，y 左=绿，z 上=蓝
void drawPoseAxes(cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec,
                  const cv::Mat& K, const cv::Mat& dist) {
    std::vector<cv::Point3f> axis = {
        {0.f, 0.f, 0.f},   // 原点 = 装甲板中心
        {60.f, 0.f, 0.f},  // x 轴：前方
        {0.f, 60.f, 0.f},  // y 轴：左侧
        {0.f, 0.f, 60.f},  // z 轴：上方
    };
    std::vector<cv::Point2f> pts;
    cv::projectPoints(axis, rvec, tvec, K, dist, pts);

    cv::line(img, pts[0], pts[1], cv::Scalar(0, 0, 255), 2);   // x 红
    cv::line(img, pts[0], pts[2], cv::Scalar(0, 255, 0), 2);   // y 绿
    cv::line(img, pts[0], pts[3], cv::Scalar(255, 0, 0), 2);   // z 蓝
    cv::circle(img, pts[0], 3, cv::Scalar(255, 255, 255), -1); // 原点白点
}

int main(int argc, char** argv) {
    bool show = argc > 1 && std::string(argv[1]) == "show";

    cv::VideoCapture cap("video/armor.avi");
    if (!cap.isOpened()) {
        std::cerr << "打不开视频 video/armor.avi（请在项目根目录 ~/light_bar 运行）" << std::endl;
        return 1;
    }

    // 相机内参和畸变矩阵（PnP 位姿解算用）
    cv::Mat K = cv::Mat(3, 3, CV_64F, const_cast<double*>(CAM_K)).clone();
    cv::Mat dist = cv::Mat(1, 5, CV_64F, const_cast<double*>(CAM_DIST)).clone();

    // 输出视频：mp4v 编码，Windows 自带的播放器也能直接打开
    cv::VideoWriter out("output.mp4",
                        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        cap.get(cv::CAP_PROP_FPS),
                        cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
                                 (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

    cv::Mat frame;
    int idx = 0, hit = 0;
    while (cap.read(frame)) {
        std::vector<LightBar> bars;
        detectLightBars(frame, bars);

        // 候选多于两个时（阈值低会带进噪声），挑最像的一对画出来
        int iL = -1, iR = -1;
        if (bars.size() >= 2 && findBestPair(bars, iL, iR)) {
            drawArmor(frame, bars[iL], bars[iR]);

            // 选做：PnP 位姿解算 + 在装甲板中心绘制坐标轴
            cv::Mat rvec, tvec;
            if (solveArmorPose(bars[iL], bars[iR], K, dist, rvec, tvec)) {
                drawPoseAxes(frame, rvec, tvec, K, dist);
                // 打印位姿和装甲板法向量（x 轴）的 z 分量：
                // 装甲板正对相机时该值应小于 0（法向量指向相机，即 -z 方向）
                cv::Mat R;
                cv::Rodrigues(rvec, R);
                double normalZ = R.at<double>(2, 0);
                std::cout << "  pose t(mm): " << tvec.t()
                          << "  normalZ=" << normalZ << std::endl;
            }
            ++hit;
        }
        std::cout << "frame " << idx << ": 灯条候选 " << bars.size() << " 个" << std::endl;

        out.write(frame);
        if (idx % 5 == 0)
            cv::imwrite("out_frames/result_" + std::to_string(idx) + ".jpg", frame);

        if (show) {
            cv::imshow("light bar", frame);
            if (cv::waitKey(30) == 27) break;   // 按 ESC 提前退出
        }
        ++idx;
    }
    std::cout << "共处理 " << idx << " 帧，成功识别装甲板 " << hit << " 帧" << std::endl;
    return 0;
}
