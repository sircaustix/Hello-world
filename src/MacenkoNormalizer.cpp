#include "MacenkoNormalizer.h"
#include "Utils.h"
#include <iostream>
#include <string>
using namespace std;
using namespace ColourStainNormalization;
std::unique_ptr<ColourStainNormalization::MacenkoNormalizer> ColourStainNormalization::MacenkoNormalizer::New()
{
    return std::make_unique<ColourStainNormalization::MacenkoNormalizer>(ColourStainNormalization::MacenkoNormalizer());
}
ColourStainNormalization::MacenkoNormalizer ColourStainNormalization::MacenkoNormalizer::init()
{
    return ColourStainNormalization::MacenkoNormalizer();
}

ColourStainNormalization::MacenkoNormalizer::MacenkoNormalizer() : MatrixMethods()
{
}

void ColourStainNormalization::MacenkoNormalizer::computeStainMatrix(cv::Mat image, Eigen::MatrixXd &_stainMatrix)
{
    cv::Mat tissueMask = Utils::getTissueMask(image);
    cv::Mat OD = Utils::convertRGBToOD(image);
    Eigen::MatrixXd C = Utils::convertToEigenFormat(OD);
    Eigen::MatrixXd D = Utils::convertToEigenFormat(tissueMask);
    // Eigen::Matrix<bool, 1, Eigen::Dynamic> non_zeros = D.cast<bool>().rowwise().any();
    // //cout << D.rows() << "X" << D.cols() << endl;
    // //cout << non_zeros.count() << endl;
    // Eigen::MatrixXd OD_E(non_zeros.count(), C.cols());
    // Eigen::Index j = 0;
    // for (Eigen::Index i = 0; i < C.rows(); ++i)
    // {
    //     if (non_zeros(i))
    //         OD_E.row(j++) = C.row(i);
    // }
    //Utils::evaluateStainMatrix(OD_E, test);
    // Eigen::MatrixXd res = A("", A.cast<bool>().colwise().any());
    //cout << "Covariance" << endl;
    Eigen::MatrixXd covariance = Utils::cov(C, false);
    Eigen::VectorXcd eigenValues;
    Eigen::MatrixXcd V = Utils::eigenDecomposition(covariance, eigenValues);
    int indices[2];
    eigenValues.real().maxCoeff(&indices[0]);
    eigenValues.real().minCoeff(&indices[1]);
    Eigen::MatrixXd Vr = V.real();
    if (indices[1] != 0)
    {
        Vr.col(0).swap(Vr.col(indices[1]));
    }
    Eigen::MatrixXd Vrs = Vr.block(0, 1, V.rows(), 2);
    if (Vrs(0, 0) < 0)
    {
        for (int i = 0; i < Vrs.rows(); i++)
        {
            Vrs(i, 0) *= -1;
        }
    }
    if (Vrs(0, 1) < 0)
    {
        for (int i = 0; i < Vrs.rows(); i++)
        {
            Vrs(i, 1) *= -1;
        }
    }
    Eigen::MatrixXd projectionMatrix = C * Vrs;
    Eigen::MatrixXd angleMatrix(projectionMatrix.rows(), 1);
    for (int i = 0; i < projectionMatrix.rows(); i++)
    {
        angleMatrix(i, 0) = atan2(projectionMatrix(i, 1), projectionMatrix(i, 0));
    }
    vector<double> sortedVector;
    for (int i = 0; i < angleMatrix.rows(); i++)
    {
        sortedVector.push_back(angleMatrix(i, 0));
    }
    std::sort(sortedVector.begin(), sortedVector.end());
    double maxPhi = Utils::computePercentile(sortedVector, 99);
    double minPhi = Utils::computePercentile(sortedVector, 1);
    Eigen::MatrixXd rotationMatrix(2, 2);
    rotationMatrix(0, 0) = cos(maxPhi);
    rotationMatrix(1, 0) = sin(maxPhi);
    rotationMatrix(0, 1) = cos(minPhi);
    rotationMatrix(1, 1) = sin(minPhi);
    Eigen::MatrixXd HE = (Vrs * rotationMatrix).transpose();
    if (HE(0, 0) < HE(1, 0))
    {
        HE.row(0).swap(HE.row(1));
    }
    _stainMatrix = HE.rowwise().normalized();
}

void ColourStainNormalization::MacenkoNormalizer::computeConcentrationMatrix(cv::Mat image, Eigen::MatrixXd &_concentrationMatrix)
{
    cv::Mat OD = Utils::convertRGBToOD(image);
    Eigen::MatrixXd C = Utils::convertToEigenFormat(OD);
    Utils::computeConcentrationMatrix(C, stainMatrix, _concentrationMatrix);
}

void ColourStainNormalization::MacenkoNormalizer::fit(cv::Mat target)
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

void ColourStainNormalization::MacenkoNormalizer::transform(cv::Mat source, cv::Mat &output)
{
    Eigen::MatrixXd _stainMatrix;
    Eigen::MatrixXd _concentrationMatrix;
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
        temp.resize(512, 512);
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
    output = cv::Mat(RGB.t());
    // cv::imwrite("Test22.jpg", RGB);
    //cout << temp.rows() << "x" << temp.cols() << endl;
}
