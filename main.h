class LicensePlate {
  public:
    void viewer(const cv::Mat& frame, std::string title);
    std::vector<std::vector<cv::Point>> locateCandidates(cv::Mat& frame);
    void drawLicensePlate(cv::Mat& frame, std::vector<std::vector<cv::Point>>& candidates);

  private:
    void grayscale(cv::Mat& frame);
};