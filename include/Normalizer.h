#pragma once
#include "Module.h"
#include <memory>
//#include <boost/optional.hpp>
class ColourStainNormalization::Normalizer
{
    friend std::default_delete<ColourStainNormalization::Normalizer>;

public:
    static std::unique_ptr<ColourStainNormalization::Normalizer> init() { return nullptr; }
    virtual void transform(cv::Mat, cv::Mat &) = 0;
    virtual void fit(cv::Mat) = 0;
    virtual void eval(cv::Mat, int mode=0){}
protected:
    Normalizer() {}
    ~Normalizer(){};
};
