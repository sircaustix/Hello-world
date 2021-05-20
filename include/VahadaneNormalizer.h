#pragma once
#include "MatrixMethods.h"
class ColourStainNormalization::VahadaneNormalizer : public ColourStainNormalization::MatrixMethods
{
    friend std::default_delete<ColourStainNormalization::VahadaneNormalizer>;

public:
    static std::unique_ptr<ColourStainNormalization::VahadaneNormalizer> New();
    static ColourStainNormalization::VahadaneNormalizer init();
    void transform(cv::Mat, cv::Mat &);
    void fit(cv::Mat);

protected:
    Eigen::MatrixXd stainMatrix;
    Eigen::MatrixXd concentrationMatrix;
    void computeStainMatrix(cv::Mat, Eigen::MatrixXd &_stainMatrix);
    void computeConcentrationMatrix(cv::Mat, Eigen::MatrixXd &_concentrationMatrix);
    VahadaneNormalizer();

private:
};