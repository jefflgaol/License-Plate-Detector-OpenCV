#include "opencv2/opencv.hpp"
#include <iostream>
#include "main.h"

#define MIN_AR 1        // Minimum aspect ratio
#define MAX_AR 6        // Maximum aspect ratio
#define KEEP 5          // Limit the number of license plates
#define RECT_DIFF 2000  // Set the difference between contour and rectangle

// Random generator for cv::Scalar
cv::RNG rng(12345);

bool compareContourAreas (std::vector<cv::Point>& contour1, std::vector<cv::Point>& contour2) {
    const double i = fabs(contourArea(cv::Mat(contour1)));
    const double j = fabs(contourArea(cv::Mat(contour2)));
    return (i < j);
}

void LicensePlate::grayscale(cv::Mat& frame) {
  cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
}

void LicensePlate::drawLicensePlate(cv::Mat& frame, std::vector<std::vector<cv::Point>>& candidates) {
  const int width = frame.cols;
  const int height = frame.rows;
  const float ratio_width = width / (float) 512;    // WARNING! Aspect ratio may affect the performance (TO DO LIST)
  const float ratio_height = height / (float) 512;  // WARNING! Aspect ratio may affect the performance

  // Convert to rectangle and decide which one violate the aspect ratio and dimension relative to image input dimension.
  /*
  std::vector<cv::Rect> non_overlapping_rect;
  for (std::vector<cv::Point> currentCandidate : candidates) {
    cv::Rect rectangle_bounding = cv::boundingRect(currentCandidate);
    float aspect_ratio = rectangle_bounding.width / (float) rectangle_bounding.height;
    if (aspect_ratio >= MIN_AR && aspect_ratio <= MAX_AR && 
        rectangle_bounding.width < 0.5 * (float) frame.cols && rectangle_bounding.height < 0.5 * (float) frame.rows)  {
      float difference = rectangle_bounding.area() - contourArea(currentCandidate);
      if (difference < RECT_DIFF) {
        non_overlapping_rect.push_back(rectangle_bounding);
      } 
    }
  }
  */

  // Convert to rectangle and also filter out the non-rectangle-shape.
  std::vector<cv::Rect> rectangles;
  for (std::vector<cv::Point> currentCandidate : candidates) {
    cv::Rect temp = cv::boundingRect(currentCandidate);
    float difference = temp.area() - cv::contourArea(currentCandidate);
    if (difference < RECT_DIFF) {
      rectangles.push_back(temp);
    }
  }

  // Remove rectangle with 好奇怪 shape.
  rectangles.erase(std::remove_if(rectangles.begin(), rectangles.end(), [](cv::Rect temp) {
    const float aspect_ratio = temp.width / (float) temp.height;
    return aspect_ratio < MIN_AR || aspect_ratio > MAX_AR;
  }), rectangles.end());

  /*
  // Find overlapping rectangle and draw it (also return to the original dimension).
  for (int i = 0; i < rectangles.size(); i++) {
    bool intersects = false;
    for (int j = i + 1; j < rectangles.size(); j++) {
      if (i == j) {
        break;
      }
      intersects = ((rectangles[i] & rectangles[j]).area() > 0);
      if (intersects) {
        break;
      }
    }
    if (!intersects) {
      cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
      cv::rectangle(frame, cv::Point(rectangles[i].x * ratio_width, rectangles[i].y * ratio_height), cv::Point((rectangles[i].x + rectangles[i].width) * ratio_width, (rectangles[i].y + rectangles[i].height) * ratio_height), color, 3, cv::LINE_8, 0);
    }
  }
  */

  /*
  rectangles.erase(std::max_element(rectangles.begin(), rectangles.end(), [](cv::Rect left, cv::Rect right) {
    return left.area() < right.area();
  }));
  */

  for (cv::Rect rectangle : rectangles) {
    cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    cv::rectangle(frame, cv::Point(rectangle.x * ratio_width, rectangle.y * ratio_height), cv::Point((rectangle.x + rectangle.width) * ratio_width, (rectangle.y + rectangle.height) * ratio_height), color, 3, cv::LINE_8, 0);
  }
}

std::vector<std::vector<cv::Point>> LicensePlate::locateCandidates(cv::Mat& frame) {
  // Reduce the image dimension to process
  cv::Mat processedFrame = frame;
  cv::resize(frame, processedFrame, cv::Size(512, 512));

  // Must be converted to grayscale
  if (frame.channels() == 3) {
    LicensePlate::grayscale(processedFrame);
  }

  // Perform blackhat morphological operation, reveal dark regions on light backgrounds
  cv::Mat blackhatFrame;
  cv::Mat rectangleKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 5)); // Shapes are set 13 pixels wide by 5 pixels tall
  cv::morphologyEx(processedFrame, blackhatFrame, cv::MORPH_BLACKHAT, rectangleKernel);

  // Find license plate based on whiteness property
  cv::Mat lightFrame;
  cv::Mat squareKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(processedFrame, lightFrame, cv::MORPH_CLOSE, squareKernel);
  cv::threshold(lightFrame, lightFrame, 0, 255, cv::THRESH_OTSU);

  // Compute Sobel gradient representation from blackhat using 32 float,
  // and then convert it back to normal [0, 255]
  cv::Mat gradX;
  double minVal, maxVal;
  int dx = 1, dy = 0, ddepth = CV_32F, ksize = -1;
  cv::Sobel(blackhatFrame, gradX, ddepth, dx, dy, ksize); // Looks coarse if imshow, because the range is high?
  gradX = cv::abs(gradX);
  cv::minMaxLoc(gradX, &minVal, &maxVal);
  gradX = 255 * ((gradX - minVal) / (maxVal - minVal));
  gradX.convertTo(gradX, CV_8U);

  // Blur the gradient result, and apply closing operation
  cv::GaussianBlur(gradX, gradX, cv::Size(5,5), 0);
  cv::morphologyEx(gradX, gradX, cv::MORPH_CLOSE, rectangleKernel);
  cv::threshold(gradX, gradX, 0, 255, cv::THRESH_OTSU);

  // Erode and dilate
  cv::erode(gradX, gradX, 2);
  cv::dilate(gradX, gradX, 2);

  // Bitwise AND between threshold result and light regions
  cv::bitwise_and(gradX, gradX, lightFrame);
  cv::dilate(gradX, gradX, 2);
  cv::erode(gradX, gradX, 1);

  // Find contours in the thresholded image and sort by size
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(gradX, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  std::sort(contours.begin(), contours.end(), compareContourAreas);
  std::vector<std::vector<cv::Point>> top_contours;
  top_contours.assign(contours.end() - KEEP, contours.end()); // Descending order

  return top_contours;
}

void LicensePlate::viewer(const cv::Mat& frame, std::string title) {
  cv::imshow(title, frame);
}

int main( int argc, char** argv ) {
  // Instantiate LicensePlate object
  LicensePlate lp;

/*
 __     __    __     ______     ______     ______    
/\ \   /\ "-./  \   /\  __ \   /\  ___\   /\  ___\   
\ \ \  \ \ \-./\ \  \ \  __ \  \ \ \__ \  \ \  __\   
 \ \_\  \ \_\ \ \_\  \ \_\ \_\  \ \_____\  \ \_____\ 
  \/_/   \/_/  \/_/   \/_/\/_/   \/_____/   \/_____/ 
*/
  std::string filename = "001.jpg";
  cv::Mat image;
  image = cv::imread(filename, cv::IMREAD_COLOR);
  if(!image.data ) {
    std::cout <<  "Image not found or unable to open" << std::endl ;
    return -1;
  }
  std::vector<std::vector<cv::Point>> candidates = lp.locateCandidates(image);
  lp.drawLicensePlate(image, candidates);
  lp.viewer(image, "Frame");
  cv::waitKey(0);

  /*
 __   __   __     _____     ______     ______    
/\ \ / /  /\ \   /\  __-.  /\  ___\   /\  __ \   
\ \ \'/   \ \ \  \ \ \/\ \ \ \  __\   \ \ \/\ \  
 \ \__|    \ \_\  \ \____-  \ \_____\  \ \_____\ 
  \/_/      \/_/   \/____/   \/_____/   \/_____/ 
  */
  /*
  cv::VideoCapture cap("demo.mp4");
  if (!cap.isOpened()) {
    std::cout << "Error opening video stream or file" << std::endl;
  }

  while (true) {
    // Image digester
    cv::Mat image;
    cap >> image;
    if (image.empty()) break;

    // Keyboard listener
    char c = (char) cv::waitKey(1);
    if (c == 200) break;

    // Processing technique
    std::vector<std::vector<cv::Point>> candidates = lp.locateCandidates(image);

    cv::Mat drawing = cv::Mat::zeros(image.size(), CV_8UC3);
    std::vector<cv::Vec4i> hierarchy;
    for (int i = 0; i < candidates.size(); i++) {
      cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
      cv::drawContours(drawing, candidates, i, color, 2, 8, hierarchy, 0, cv::Point() );
    }
    cv::imshow("Drawing", drawing);
    cv::waitKey(0);

    lp.drawLicensePlate(image, candidates);
    lp.viewer(image, "Frame");
  }
  cap.release();
  cv::destroyAllWindows();
  */

  return 0;
}

/*
// Debug code for drawing contours
cv::Mat drawing = cv::Mat::zeros(image.size(), CV_8UC3);
std::vector<cv::Vec4i> hierarchy;
for (int i = 0; i < candidates.size(); i++) {
  cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
  cv::drawContours(drawing, candidates, i, color, 2, 8, hierarchy, 0, cv::Point() );
}
cv::imshow("Drawing", drawing);
cv::waitKey(0);
*/