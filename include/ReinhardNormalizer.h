#pragma once
#include "Normalizer.h"

class ColourStainNormalization::ReinhardNormalizer : public ColourStainNormalization::Normalizer
{
    friend std::default_delete<ColourStainNormalization::ReinhardNormalizer>;

public:
    static std::unique_ptr<ColourStainNormalization::Normalizer> New();
    static ColourStainNormalization::ReinhardNormalizer init() { return ReinhardNormalizer(); }
    void transform(cv::Mat source, cv::Mat &output);
    void fit(cv::Mat target);

protected:
    explicit ReinhardNormalizer() : Normalizer() {}
    //~ReinhardNormalizer() {}
    static ColourStainNormalization::ReinhardNormalizer normalizer;

private:
    void labSplit(cv::Mat I, cv::Mat &I1, cv::Mat &I2, cv::Mat &I3);
    cv::Mat mergeBack(cv::Mat I1, cv::Mat I2, cv::Mat I3);
    void getMeanStd(cv::Mat I, cv::Scalar &mean, cv::Scalar &std);
    cv::Scalar targetMean;
    cv::Scalar targetStd;
};