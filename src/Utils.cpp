#include <iostream>
#include "Utils.h"
#include <NumCpp.hpp>
#include <spams.h>
#include <omp.h>
//#include <NumCpp.hpp>
using namespace std;
const double ColourStainNormalization::Utils::THRESHOLD = 1e-6;
template <typename T>
SpMatrix<T> *cppLasso(Matrix<T> &X, Matrix<T> &D, Matrix<T> **path, bool return_reg_path,
                      int L, const T constraint, const T lambda2 = 0., constraint_type mode = PENALTY,
                      const bool pos = true, const bool ols = false, const int numThreads = -1,
                      int max_length_path = -1, const bool verbose = false, bool cholevsky = false)
{
    return _lassoD(&X, &D, path, return_reg_path, L, constraint, lambda2, mode, pos, ols, numThreads,
                   max_length_path, verbose, cholevsky);
}

Eigen::MatrixXd ColourStainNormalization::Utils::cov(Eigen::MatrixXd m, bool rowvar)
{
    if (rowvar)
    {
        Eigen::MatrixXd centered = (m.colwise() - m.rowwise().mean());
        Eigen::MatrixXd covariance = (centered * centered.adjoint()) / double(m.rows() - 1);
        //cout << covariance << endl;
        return covariance;
    }
    Eigen::MatrixXd centered = (m.rowwise() - m.colwise().mean());
    Eigen::MatrixXd covariance = (centered.adjoint() * centered) / double(m.rows() - 1);
    return covariance;
}

Eigen::MatrixXcd ColourStainNormalization::Utils::eigenDecomposition(Eigen::MatrixXd A, Eigen::VectorXcd &eigenValues)
{
    Eigen::EigenSolver<Eigen::MatrixXd> es(A);
    eigenValues = Eigen::VectorXcd(es.eigenvalues());
    return es.eigenvectors();
}

cv::Mat ColourStainNormalization::Utils::convertRGBToOD(cv::Mat I)
{
    cv::Mat I_Double = cv::Mat::zeros(I.size(), CV_64FC3);
    cv::Mat mask = cv::Mat::zeros(I.size(), I.type());
    cv::Mat diff;
    cv::compare(I, mask, diff, cv::CmpTypes::CMP_EQ);
    diff = diff / 255;
    cv::Mat OD = I + diff;
    OD.convertTo(OD, CV_64FC3);
    //cout << OD.type() << endl;
    OD /= 255.0;
    cv::log(OD, OD);
    OD *= -1;
    diff.release();
    mask.release();
    mask = cv::Mat(OD.size(), OD.type(), cv::Scalar(THRESHOLD, THRESHOLD, THRESHOLD));
    cv::compare(OD, mask, diff, cv::CmpTypes::CMP_LT);
    diff.convertTo(diff, CV_64FC3);
    diff /= 255.0;
    diff *= THRESHOLD;
    OD += diff;
    mask.release();
    diff.release();
    return OD;
}
void ColourStainNormalization::Utils::convertRGBToOD(cv::Mat I, cv::Mat &OD)
{
    cv::Mat I_Double = cv::Mat::zeros(I.size(), CV_64FC3);
    cv::Mat mask = cv::Mat::zeros(I.size(), I.type());
    cv::Mat diff;
    cv::compare(I, mask, diff, cv::CmpTypes::CMP_EQ);
    diff = diff / 255;
    OD = I + diff;
    OD.convertTo(OD, CV_64FC3);
    //cout << OD.type() << endl;
    OD /= 255.0;
    cv::log(OD, OD);
    OD *= -1;
    diff.release();
    mask.release();
    mask = cv::Mat(OD.size(), OD.type(), cv::Scalar(THRESHOLD, THRESHOLD, THRESHOLD));
    cv::compare(OD, mask, diff, cv::CmpTypes::CMP_LT);
    diff.convertTo(diff, CV_64FC3);
    diff /= 255.0;
    diff *= THRESHOLD;
    OD += diff;
    mask.release();
    diff.release();
}
cv::Mat ColourStainNormalization::Utils::convertODToRGB(cv::Mat OD)
{
    double min, max;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(OD, &min, &max, &minLoc, &maxLoc);
    if (min < 0.0)
    {
        throw std::runtime_error("Negative Values in OD Matrix");
    }
    cv::Mat mask = cv::Mat::zeros(OD.size(), OD.type());
    cv::Mat diff;
    mask = cv::Mat(OD.size(), OD.type(), cv::Scalar(THRESHOLD, THRESHOLD, THRESHOLD));
    cv::compare(OD, mask, diff, cv::CmpTypes::CMP_LT);
    diff.convertTo(diff, CV_64FC3);
    diff /= 255.0;
    diff *= THRESHOLD;
    OD += diff;
    OD *= -1;
    cv::exp(OD, OD);
    OD *= 255.0;
    cv::Mat RGB;
    OD.convertTo(RGB, CV_8UC3);
    return RGB;
}
cv::Mat ColourStainNormalization::Utils::getTissueMask(cv::Mat I, double luminosity_threshold)
{
    cv::Mat I_LAB;
    cv::cvtColor(I, I_LAB, cv::COLOR_RGB2Lab);
    vector<cv::Mat> LAB;
    cv::split(I_LAB, LAB);
    cv::Mat LAB_L = LAB[0];
    LAB_L.convertTo(LAB_L, CV_64FC3);
    LAB_L /= 255.0;
    Eigen::MatrixXd test_mat;

    cv::Mat mask(LAB_L.size(), LAB_L.type(), cv::Scalar(luminosity_threshold));

    cv::Mat diff;
    cv::compare(LAB_L, mask, diff, cv::CmpTypes::CMP_LT);
    diff /= 255;
    cv::Mat tissueMask;
    vector<cv::Mat> channels;
    channels.push_back(diff);
    channels.push_back(diff);
    channels.push_back(diff);
    cv::merge(channels, tissueMask);
    tissueMask.convertTo(tissueMask, CV_64FC3);
    return diff;
}

Eigen::MatrixXd ColourStainNormalization::Utils::convertToEigenFormat(cv::Mat image)
{
    int height = image.rows;
    int width = image.cols;
    vector<cv::Mat> channels;
    cv::split(image, channels);
    vector<Eigen::MatrixXd> eigen(channels.size());
    if (channels.size() == 1)
    {
        cv::cv2eigen(channels[0], eigen[0]);
        eigen[0].resize(width * height, 1);
        return eigen[0];
    }

    for (int i = 0; i < static_cast<int>(channels.size()); i++)
    {
        cv::cv2eigen(channels[i], eigen[i]);
    }
    for (int i = 0; i < static_cast<int>(eigen.size()); i++)
    {
        eigen[i].transposeInPlace();
        eigen[i].resize(width * height, 1);
    }
    Eigen::MatrixXd C(eigen[0].rows(), eigen[0].cols() + eigen[1].cols() + eigen[2].cols());
    C << eigen[0], eigen[1], eigen[2];
    C.resize(width * height, static_cast<int>(channels.size()));
    return C;
}
double ColourStainNormalization::Utils::computePercentile(std::vector<double> sortedVector, float percentile)
{
    // unsigned int length = sortedVector.size();
    // float index = (percentile * length) / 100;
    // double *sortedArray = sortedVector.data();
    // return sortedArray[static_cast<int>(index)];
    // float length = sortedVector.size();
    // float index = (percentile * length) / 100;
    // double *sortedArray = sortedVector.data();
    // double val_lower = sortedArray[static_cast<int>(index)];
    // double val_higher = sortedArray[static_cast<int>(index)+1];
    // double interp = (val_higher-val_lower)*(index-static_cast<int>(index));  
    // return val_lower+interp;
    unsigned int length = sortedVector.size();
    double *sortedArray = sortedVector.data();
    nc::NdArray<double>test_array(sortedArray, length, false);
	//test_array.print();
	//nc::Axis::NONE/ROW/COL
	nc::NdArray<double> _percentile = nc::percentile(test_array, percentile);
	//percentile.print();
    return _percentile[0];
    // return 0.0;
}

bool ColourStainNormalization::Utils::checkIfImageValid(cv::Mat image)
{
    if (!image.data)
    {
        throw std::runtime_error("Empty Input Image!");
    }
    if (image.type() != CV_8UC3)
    {
        throw std::runtime_error("Not an RGB Image!");
    }
    return true;
}

void ColourStainNormalization::Utils::computeConcentrationMatrix(Eigen::MatrixXd C, Eigen::MatrixXd stainMatrix, Eigen::MatrixXd &concentrationMatrix)
{

    int m = C.rows();
    int n = C.cols();
    double *data = C.data();
    Matrix<double> X(data, m, n);
    double *data2 = stainMatrix.data();
    m = stainMatrix.rows();
    n = stainMatrix.cols();
    Matrix<double> D(data2, m, n);
    Matrix<double> Xt;
    X.transpose(Xt);
    Matrix<double> Dt;
    D.transpose(Dt);
    Matrix<double> *path = NULL;
    // spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True).toarray().T
    SpMatrix<double> *spa = cppLasso(Xt, Dt, &path, false, -1, 0.01);
    // def lasso(X,D= None,Q = None,q = None,return_reg_path = False,L= -1,lambda1= None,lambda2= 0.,
    //              mode= spams_wrap.PENALTY,pos= False,ols= False,numThreads= -1,
    //              max_length_path= -1,verbose=False,cholesky= False):
                   
    Matrix<double> alpha;
    spa->toFull(alpha);
    m = alpha.m();
    n = alpha.n();
    double *datac = alpha.rawX();
    Eigen::Map<Eigen::MatrixXd> M(datac, m, n);
    concentrationMatrix = M.transpose();
    delete spa;
    //cout << concentrationMatrix << endl;
}
void ColourStainNormalization::Utils::evaluateStainMatrix(Eigen::MatrixXd X, Eigen::MatrixXd &_stainMatrix)
{
    ParamDictLearn<double> params;
    params.lambda = 0.1;
    params.tol = 0.000001;
    params.posAlpha = true;
    params.posD = true;
    params.modeD = L2;
    params.verbose = false;
    params.lambda2 = 10e-10;
    params.batch = false;
    params.lambda3 = 0;
    params.iter = -1;
    params.fixed_step = true;
    params.ista = false;
    params.t0 = 1e-5;
    params.mode = PENALTY;
    params.regul = FISTA::NONE;
    params.expand = false;
    params.whiten = false;
    params.clean = true;
    params.gamma1 = 0;
    params.gamma2 = 0;
    params.rho = 1.0;
    params.iter_updateD = 1;
    params.stochastic = false;
    params.modeParam = static_cast<mode_compute>(0);
    params.log = false;
    params.batch = false;
    int m = X.rows();
    int n = X.cols();
    Matrix<double> Xm(X.data(), m, n);
    Matrix<double> Xmt;
    Xm.transpose(Xmt);
    int K = 2;
    //int num_threads = 16;
    //#ifdef _OPENMP
    //   num_threads = omp_get_num_procs();
    //#endif
    //int batch_size = 256 * (num_threads + 1);
    //cout << "Number of Threads = " << num_threads << endl;
    //cout << "Batch size = " << batch_size << endl;
    Trainer<double> t(K); // batch_size, num_threads);
    t.train_fista(Xmt, params);
    Matrix<double> D;
    t.getD(D);
    Matrix<double> Dt;
    D.transpose(Dt);
    //Dt.print("Test");
    m = Dt.m();
    n = Dt.n();
    double *datac = Dt.rawX();
    Eigen::Map<Eigen::MatrixXd> M(datac, m, n);
    if (M(0, 0) < M(1, 0))
    {
        M.row(0).swap(M.row(1));
    }
    _stainMatrix = M.rowwise().normalized();
}
