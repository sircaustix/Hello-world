#include "VahadaneNormalizer.h"
#include "Utils.h"
#include <iostream>
#include <string>
using namespace std;
using namespace ColourStainNormalization;
std::unique_ptr<ColourStainNormalization::VahadaneNormalizer> ColourStainNormalization::VahadaneNormalizer::New()
{
    return std::make_unique<ColourStainNormalization::VahadaneNormalizer>(ColourStainNormalization::VahadaneNormalizer());
}
ColourStainNormalization::VahadaneNormalizer ColourStainNormalization::VahadaneNormalizer::init()
{
    return ColourStainNormalization::VahadaneNormalizer();
}

ColourStainNormalization::VahadaneNormalizer::VahadaneNormalizer() : MatrixMethods()
{
}

void ColourStainNormalization::VahadaneNormalizer::computeStainMatrix(cv::Mat image, Eigen::MatrixXd &_stainMatrix)
{
    cv::Mat tissueMask = Utils::getTissueMask(image);
    // int m = 10, p = 10;
    // Matrix<double> D2(m, p);
    cv::Mat OD = Utils::convertRGBToOD(image);
    Eigen::MatrixXd C = Utils::convertToEigenFormat(OD);

    Eigen::MatrixXd D = Utils::convertToEigenFormat(tissueMask);
    Eigen::Matrix<bool, 1, Eigen::Dynamic> non_zeros = D.cast<bool>().rowwise().any();
    // Eigen::MatrixXd OD_E(non_zeros.count(), C.cols());
    // Eigen::Index j = 0;
    // for (Eigen::Index i = 0; i < C.rows(); ++i)
    // {
    //     if (non_zeros(i))
    //         OD_E.row(j++) = C.row(i);
    // }
    // Eigen::MatrixXd res = A("", A.cast<bool>().colwise().any());
    Utils::evaluateStainMatrix(C, _stainMatrix);
}

void ColourStainNormalization::VahadaneNormalizer::computeConcentrationMatrix(cv::Mat image, Eigen::MatrixXd &_concentrationMatrix)
{
    cv::Mat OD = Utils::convertRGBToOD(image);
    Eigen::MatrixXd C = Utils::convertToEigenFormat(OD);
    Utils::computeConcentrationMatrix(C, stainMatrix, _concentrationMatrix);
}

void ColourStainNormalization::VahadaneNormalizer::fit(cv::Mat target)
{
    computeStainMatrix(target, stainMatrix);
    computeConcentrationMatrix(target, concentrationMatrix);
    //Ready with target stain and computation matrices.
    int index[2];
    maxCT[0] = concentrationMatrix.col(0).maxCoeff(&index[0]);
    maxCT[1] = concentrationMatrix.col(1).maxCoeff(&index[1]);
    double *data = concentrationMatrix.col(0).data();
    vector<double> columnVectors(data, data + concentrationMatrix.rows());
    std::sort(columnVectors.begin(), columnVectors.end());
    maxCT[0] = Utils::computePercentile(columnVectors, 99);
    data = concentrationMatrix.col(1).data();
    columnVectors.erase(columnVectors.begin(), columnVectors.end());
    vector<double> columnVectors2(data, data + concentrationMatrix.rows());
    std::sort(columnVectors2.begin(), columnVectors2.end());
    maxCT[1] = Utils::computePercentile(columnVectors2, 99);
    cv::Mat stainMatrixOD;
    cv::eigen2cv(stainMatrix, stainMatrixOD);
    cv::Mat rgbStainMatrix = Utils::convertODToRGB(stainMatrixOD);
}

void ColourStainNormalization::VahadaneNormalizer::transform(cv::Mat source, cv::Mat &output)
{
    Eigen::MatrixXd _stainMatrix;
    Eigen::MatrixXd _concentrationMatrix;
    //cv::imshow("Test", source);
    //cv::waitKey(-1);
    computeStainMatrix(source, _stainMatrix);
    computeConcentrationMatrix(source, _concentrationMatrix);
    // cout << _concentrationMatrix << endl;
    double *data = _concentrationMatrix.col(0).data();
    vector<double> columnVectors(data, data + concentrationMatrix.rows());
    std::sort(columnVectors.begin(), columnVectors.end());
    double maxC[2];
    //maxC[0] = concentrationMatrix.col(0).maxCoeff(&index[0]);
    //maxC[1] = concentrationMatrix.col(1).maxCoeff(&index[1]);
    maxC[0] = Utils::computePercentile(columnVectors, 99);
    data = concentrationMatrix.col(1).data();
    columnVectors.erase(columnVectors.begin(), columnVectors.end());
    vector<double> columnVectors2(data, data + concentrationMatrix.rows());
    std::sort(columnVectors2.begin(), columnVectors2.end());
    maxC[1] = Utils::computePercentile(columnVectors2, 99);
    //cout << maxC[1] << endl;
    for (int i = 0; i < _concentrationMatrix.rows(); i++)
    {
        _concentrationMatrix(i, 0) *= (maxCT[0] / maxC[0]);
        _concentrationMatrix(i, 1) *= (maxCT[1] / maxC[1]);
    }
    //Convert to output

    vector<Eigen::MatrixXd> eigens;
    Eigen::MatrixXd _output = _concentrationMatrix * stainMatrix;
    for (int i = 0; i < _output.cols(); i++)
    {
        Eigen::MatrixXd temp = _output.col(i);
        temp.resize(source.rows, source.cols);
        eigens.push_back(temp);
    }
    vector<cv::Mat> channels(eigens.size());
    for (int i = 0; i < _output.cols(); i++)
    {
        cv::eigen2cv(eigens[i], channels[i]);
        channels[i] *= -1;
        cv::exp(channels[i], channels[i]);
        channels[i] *= 255.0;
        channels[i].convertTo(channels[i], CV_8U);
    }
    cv::Mat RGB;
    cv::merge(channels, RGB);
    RGB.convertTo(RGB, CV_8UC3);
    output = RGB.t();
}
