/**
 * @file Utils.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-05-29
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#pragma once

#include "Module.h"
/**
 * @brief Class that contains utility methods required by 
 * the other classes.
 */
class ColourStainNormalization::Utils
{
public:
    /**
     * @brief The method to compute the covariance matrix of the input matrix.
     * 
     * @param inputMatrix, the input matrix. 
     * @param rowvar, decides the mode of computation of convariance matrix. Default value is true for row variance. 
     * @return Eigen::MatrixXd, the covariance matrix. 
     */
    static Eigen::MatrixXd cov(Eigen::MatrixXd inputMatrix, bool rowvar = true);
    /**
     * @brief The method to perform the eigen decomposition of the input matrix.
     * 
     * @param inputMatrix, the input matrix. 
     * @param eigenValues, vector containing the eigen values. 
     * @return Eigen::MatrixXcd, the matrix containing the eigen vectors. 
     */
    static Eigen::MatrixXcd eigenDecomposition(Eigen::MatrixXd inputMatrix, Eigen::VectorXcd &eigenValues);
    static cv::Mat convertRGBToOD(cv::Mat);
    static void convertRGBToOD(cv::Mat, cv::Mat &);
    static cv::Mat convertODToRGB(cv::Mat);
    static cv::Mat getTissueMask(cv::Mat, double luminosity_threshold = 0.8);
    static Eigen::MatrixXd convertToEigenFormat(cv::Mat);
    static double computePercentile(std::vector<double> sortedVector, float percentile);
    static bool checkIfImageValid(cv::Mat);
    static void computeConcentrationMatrix(Eigen::MatrixXd C, Eigen::MatrixXd stainMatrix, Eigen::MatrixXd &concentrationMatrix);
    static void evaluateStainMatrix(Eigen::MatrixXd, Eigen::MatrixXd &);
    static void standardiseLuminosity(const cv::Mat input, cv::Mat &output, float percentile = 95);

private:
    Utils();
    ~Utils();
    static const double THRESHOLD;
};