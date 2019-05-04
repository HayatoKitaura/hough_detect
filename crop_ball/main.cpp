#include "detector/detector_yolo.h"
#include <experimental/filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
namespace fs = std::experimental::filesystem;

void detectBall(DetectorYOLO &dyolo, cv::Mat &src, struct BoundingBox &ball_v);
void cropBall(cv::Mat &src, cv::Mat &dst, struct BoundingBox &bbox);
void make_labelimage(cv::Mat &src, cv::Mat &dst, std::string color_table_path);
void separate_labelimage(int colorflag, cv::Mat &labelimage, cv::Mat &dst);

int main(int argc, char** argv){
    // initialize
    std::cout << "init" << std::endl;
    int img_w = 640;
    int img_h = 480;
    DetectorYOLO dyolo(img_w, img_h);
    std::string color_table = "color_table.cnf";
    std::string cfg_path    = "v2tiny_4class.cfg";
    std::string weight_path = "v2tiny_4class_backup.weights";
    std::string img_dir     = argv[1];
    std::vector<int> threshold{20,20,20,20};
    dyolo.setupYOLO(cfg_path,weight_path,threshold);
    // detect
    fs::path output_dir = "crop_ball";
    fs::create_directory(output_dir);
    for(const fs::directory_entry &i : fs::directory_iterator(img_dir)){
        if(i.path().extension().string() == ".png"  || i.path().extension().string() == ".jpg")
        {
            std::string img_path = i.path().string();
            std::cout << img_path << std::endl;
            cv::Mat img = cv::imread(img_path);
            struct BoundingBox ball_v;
            detectBall(dyolo,img,ball_v);

            cv::Mat label_16u;
            make_labelimage(img, label_16u, color_table);
            cv::Mat label;
            separate_labelimage(COLOR_GREEN, label_16u, label);
            cv::Mat c_ball;
            cropBall(label,c_ball,ball_v);
            //add
            cv::Mat d_ball;
            cv::medianBlur(c_ball, d_ball, 5);
            cv::resize(d_ball, d_ball, cv::Size(100, 100));
            //cv::GaussianBlur(d_ball, d_ball. Size(1,1), 0, 0)
            std::vector<cv::Vec3f> circles;
            cv::HoughCircles(d_ball, circles, CV_HOUGH_GRADIENT, 2, 100, 20, 10, 0, 400);
            if(circles.size() == 0)
            {
                std::cout << "skip" << std::endl;
                continue;
            }
            cv::circle(d_ball, cv::Point(circles[0][0], circles[0][1]), circles[0][2], cv::Scalar(0, 0, 225), 2, 8, 0);
            cv::circle(d_ball, cv::Point(circles[0][0], circles[0][1]), 1, cv::Scalar(0, 255, 0), 1, 4, 0);
            std::cout << "success" << std::endl;
            std::string write_name=output_dir.string() + "/" + i.path().filename().string();
            cv::imwrite(write_name,c_ball);
            cv::imshow("detect", d_ball);
            cv::imshow("cut",c_ball);
            cv::imshow("src",img);
            cv::imshow("label",label);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }

    return 0;
}

void detectBall(DetectorYOLO &dyolo, cv::Mat &src, struct BoundingBox &out){
    dyolo.setImage(src);
    std::vector<object_pos> objects;
    std::vector<object_pos> white_line;
    dyolo.getObjects(objects, white_line);
    std::vector<struct BoundingBox> ball_v = dyolo.getBallBoundingBox();
    out.x = ball_v[0].x;
    out.y = ball_v[0].y;
    out.w = ball_v[0].w;
    out.h = ball_v[0].h;
}

void cropBall(cv::Mat &src, cv::Mat &dst, struct BoundingBox &bbox){
    int X   = bbox.x * src.cols;
    int Y   = bbox.y * src.rows;
    int W   = bbox.w * src.cols;
    int H   = bbox.h * src.rows;
    cv::Mat img_cut(src,cv::Rect(X-W/2, Y-H/2,W,H));
    dst = img_cut;
}

void make_labelimage(cv::Mat &src, cv::Mat &dst, std::string color_table_path){
    cv::Mat src_ycrcb;
    cv::cvtColor(src,src_ycrcb,CV_BGR2YCrCb);
    ColorTable color_table;
    color_table.loadColorTable(color_table_path);
    cv::Mat labeling_image(src_ycrcb.rows, src_ycrcb.cols, CV_16UC1);
    color_table.apply(src_ycrcb, labeling_image);
    dst = labeling_image;
}

void separate_labelimage(int colorflag, cv::Mat &labelimage, cv::Mat &dst){
    cv::Mat mat(labelimage.rows, labelimage.cols, CV_8UC1);
    for(int x=0;x<labelimage.cols;x++)
    {
        for(int y=0;y<labelimage.rows;y++)
        {
            if( 1<<colorflag & labelimage.data[y * labelimage.cols + x])
            {
                mat.data[y * labelimage.cols + x] = 255;
            }
            else{
                mat.data[y * labelimage.cols + x] = 0;
            }
        }
    }
    dst = mat;
}
