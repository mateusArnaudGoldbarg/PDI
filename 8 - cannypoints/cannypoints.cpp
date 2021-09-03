#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <cstdlib>

using namespace std;
using namespace cv;

#define STEP 5
#define JITTER 3
#define RAIO 3

int top_slider = 10;
int top_slider_max = 200;
int step_slider = 5;
int step_slider_max = 10;
int jitter_slider = 3;
int jitter_slider_max = 10;
int raio_slider = 3;
int raio_slider_max = 10;
int width, height;

char TrackbarName[50];

Mat image, border, points;


void pointillism(int, void*) {
    vector<int> yrange;
    vector<int> xrange;

    int width, height, gray;
    int x, y;

    width = image.size().width;
    height = image.size().height;

    xrange.resize(height / STEP);
    yrange.resize(width / STEP);

    iota(xrange.begin(), xrange.end(), 0);
    iota(yrange.begin(), yrange.end(), 0);

    for (uint i = 0; i < xrange.size(); i++) {
        xrange[i] = xrange[i] * STEP + STEP / 2;
    }

    for (uint i = 0; i < yrange.size(); i++) {
        yrange[i] = yrange[i] * STEP + STEP / 2;
    }

    points = Mat(height, width, CV_8U, Scalar(255));

    random_shuffle(xrange.begin(), xrange.end());

    for (auto i : xrange) {
        random_shuffle(yrange.begin(), yrange.end());
        for (auto j : yrange) {
            x = i + rand() % (2 * JITTER) - JITTER + 1;
            y = j + rand() % (2 * JITTER) - JITTER + 1;
            gray = image.at<uchar>(x, y);
            circle(points,
                cv::Point(y, x),
                RAIO,
                CV_RGB(gray, gray, gray),
                -1,
                LINE_AA);
        }
    }
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    for (int i = 0; i < int(contours.size()); i++) {
        for (int j = 0; j < int(contours[i].size()); j++) {
            gray = image.at<uchar>(contours[i][j].y, contours[i][j].x);
            circle(points, cv::Point(contours[i][j].x, contours[i][j].y),
                1,
                CV_RGB(gray, gray, gray),
                -1,
                LINE_AA);
        }
    }

    imshow("cannypoints", points);
    imwrite("cannypoints.png", points);
}

void on_trackbar_canny(int, void*) {
    Canny(image, border, top_slider, 3 * top_slider);
    pointillism(top_slider, 0);
}

int main(int argc, char** argv) {
    image = imread("we.jpeg", IMREAD_GRAYSCALE);
    resize(image,image,Size(600,800));
    width = image.size().width;
    height = image.size().height;

    namedWindow("canny", 1);

    sprintf(TrackbarName, "Threshold inferior");
    createTrackbar(TrackbarName, "canny",
        &top_slider,
        top_slider_max,
        on_trackbar_canny);

    sprintf(TrackbarName, "step");
    createTrackbar(TrackbarName, "canny",
        &step_slider,
        step_slider_max,
        pointillism);

    sprintf(TrackbarName, "jitter");
    createTrackbar(TrackbarName, "canny",
        &jitter_slider,
        jitter_slider_max,
        pointillism);

    sprintf(TrackbarName, "raio");
    createTrackbar(TrackbarName, "canny",
        &raio_slider,
        raio_slider_max,
        pointillism);

    on_trackbar_canny(top_slider, 0);

    waitKey();
    imwrite("original.png", image);
    imwrite("cannyborders.png", border);
    return 0;
}
