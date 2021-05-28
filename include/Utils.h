#pragma once

#include "Module.h"
class ColourStainNormalization::Utils
{
public:
    static Eigen::MatrixXd cov(Eigen::MatrixXd, bool rowvar = true);
    static Eigen::MatrixXcd eigenDecomposition(Eigen::MatrixXd, Eigen::VectorXcd &eigenValues);
    static cv::Mat convertRGBToOD(cv::Mat);
    static void convertRGBToOD(cv::Mat, cv::Mat &);
    static cv::Mat convertODToRGB(cv::Mat);
    static cv::Mat getTissueMask(cv::Mat, double luminosity_threshold = 0.8);
    static Eigen::MatrixXd convertToEigenFormat(cv::Mat);
    static double computePercentile(std::vector<double> sortedVector, float percentile);
    static bool checkIfImageValid(cv::Mat);
    static void computeConcentrationMatrix(Eigen::MatrixXd C, Eigen::MatrixXd stainMatrix, Eigen::MatrixXd &concentrationMatrix);
    static void evaluateStainMatrix(Eigen::MatrixXd, Eigen::MatrixXd &);

private:
    Utils();
    ~Utils();
    static const double THRESHOLD;
};