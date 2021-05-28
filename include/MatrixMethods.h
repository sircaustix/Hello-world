#pragma once
#include "Normalizer.h"
#include <iostream>
class ColourStainNormalization::MatrixMethods : public ColourStainNormalization::Normalizer
{
    friend std::default_delete<ColourStainNormalization::MatrixMethods>;

public:
protected:
    Eigen::MatrixXd stainMatrix;
    Eigen::MatrixXd concentrationMatrix;
    double maxCT[2] = {0.0, 0.0};
    virtual void computeStainMatrix(cv::Mat, Eigen::MatrixXd &_stainMatrix) = 0;
    virtual void computeConcentrationMatrix(cv::Mat, const Eigen::MatrixXd _stainMatrix,Eigen::MatrixXd &_concentrationMatrix) = 0;
    MatrixMethods() : Normalizer() {}
};