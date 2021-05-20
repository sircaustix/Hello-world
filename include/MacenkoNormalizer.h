#pragma once
#include "MatrixMethods.h"
class ColourStainNormalization::MacenkoNormalizer : public ColourStainNormalization::MatrixMethods
{
    friend std::default_delete<ColourStainNormalization::MacenkoNormalizer>;

public:
    static std::unique_ptr<ColourStainNormalization::MacenkoNormalizer> New();
    static ColourStainNormalization::MacenkoNormalizer init();
    void transform(cv::Mat, cv::Mat &);
    void fit(cv::Mat);

protected:
    Eigen::MatrixXd stainMatrix;
    Eigen::MatrixXd concentrationMatrix;
    void computeStainMatrix(cv::Mat, Eigen::MatrixXd &_stainMatrix);
    void computeConcentrationMatrix(cv::Mat, Eigen::MatrixXd &_concentrationMatrix);
    MacenkoNormalizer();
    //~MacenkoNormalizer();

private:
};