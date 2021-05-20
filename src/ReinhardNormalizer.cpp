#include "ReinhardNormalizer.h"
#include "Utils.h"
#include <iostream>
using namespace std;

std::unique_ptr<ColourStainNormalization::Normalizer> ColourStainNormalization::ReinhardNormalizer::New()
{
    return make_unique<ColourStainNormalization::ReinhardNormalizer>(ColourStainNormalization::ReinhardNormalizer());
}

void ColourStainNormalization::ReinhardNormalizer::fit(cv::Mat target)
{
    getMeanStd(target, targetMean, targetStd);
    cerr << targetMean[0] << " " << targetMean[1] << " " << targetMean[2] << endl;
    cerr << targetStd[0] << " " << targetStd[1] << " " << targetStd[2] << endl;
}

void ColourStainNormalization::ReinhardNormalizer::transform(cv::Mat I, cv::Mat &output)
{
    cv::Mat I1, I2, I3;
    labSplit(I, I1, I2, I3);
    cv::Scalar means, stds;
    getMeanStd(I, means, stds);
    cv::Mat norm1 = ((I1 - means[0]) * (targetStd[0] / stds[0])) + targetMean[0];
    cv::Mat norm2 = ((I2 - means[1]) * (targetStd[1] / stds[1])) + targetMean[1];
    cv::Mat norm3 = ((I3 - means[2]) * (targetStd[2] / stds[2])) + targetMean[2];
    output = mergeBack(norm1, norm2, norm3);
}

void ColourStainNormalization::ReinhardNormalizer::labSplit(cv::Mat I, cv::Mat &I1, cv::Mat &I2, cv::Mat &I3)
{
    try
    {
        Utils::checkIfImageValid(I);
    }
    catch (std::exception &e)
    {
        throw e;
    }

    cv::Mat I_LAB(I.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    cv::cvtColor(I, I_LAB, cv::COLOR_RGB2Lab);
    vector<cv::Mat> channels;
    cv::split(I_LAB, channels);
    I1 = channels[0] / 2.55;  //should now be in range[0, 100]
    I2 = channels[1] - 128.0; //should now be in range[-127, 127]
    I3 = channels[2] - 128.0; //should now be in range[-127, 127]
}

cv::Mat ColourStainNormalization::ReinhardNormalizer::mergeBack(cv::Mat I1, cv::Mat I2, cv::Mat I3)
{
    I1 *= 2.55;  //should now be in range[0, 255]
    I2 += 128.0; // #should now be in range[0, 255]
    I3 += 128.0; //should now be in range[0, 255]
    vector<cv::Mat> channels;
    channels.push_back(I1);
    channels.push_back(I2);
    channels.push_back(I3);
    cv::Mat I;
    cv::merge(channels, I);
    I.convertTo(I, CV_8UC3);
    return I;
}

void ColourStainNormalization::ReinhardNormalizer::getMeanStd(cv::Mat I, cv::Scalar &mean, cv::Scalar &std)
{
    try
    {
        Utils::checkIfImageValid(I);
    }
    catch (std::exception &e)
    {
        throw e;
    }
    cv::Mat I1, I2, I3;
    labSplit(I, I1, I2, I3);
    cv::Scalar m1, sd1, m2, sd2, m3, sd3;
    cv::meanStdDev(I1, m1, sd1);
    cv::meanStdDev(I2, m2, sd2);
    cv::meanStdDev(I3, m3, sd3);
    mean = cv::Scalar(m1[0], m2[0], m3[0]);
    std = cv::Scalar(sd1[0], sd2[0], sd3[0]);
}