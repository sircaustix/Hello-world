#pragma once
#include "MatrixMethods.h"
class ColourStainNormalization::MacenkoNormalizer : public ColourStainNormalization::MatrixMethods
{

public:
    static std::unique_ptr<ColourStainNormalization::MacenkoNormalizer> New();
    static ColourStainNormalization::MacenkoNormalizer init();
    void transform(cv::Mat, cv::Mat &);
    void fit(cv::Mat);

protected:
    Eigen::MatrixXd stainMatrix;
    Eigen::MatrixXd concentrationMatrix;
    void computeStainMatrix(cv::Mat, Eigen::MatrixXd &_stainMatrix);
    void computeConcentrationMatrix(cv::Mat, const Eigen::MatrixXd _stainMatrix, Eigen::MatrixXd &_concentrationMatrix);
    MacenkoNormalizer();

private:
};