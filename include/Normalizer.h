#pragma once
#include "Module.h"
#include <memory>
/**
 * @brief The base class from which all stain normalization methods are derived.
 * 
 */
class ColourStainNormalization::Normalizer
{
    //Requirement for invoking unique_ptr, a smart pointer, that handles
    //memory management.
    friend std::default_delete<ColourStainNormalization::Normalizer>;

public:
    /**
     * @brief Static Method to create a unique pointer.
     * 
     * @return std::unique_ptr<ColourStainNormalization::Normalizer> 
     */
    static std::unique_ptr<ColourStainNormalization::Normalizer> New() { return nullptr; }
    /**
     * @brief Method to transform the colours of an input image to that of
     * a target image.
     * @param cv::Mat inputImage, the image to be transformed.
     * @param outputImage, the transformed image.
     */
    virtual void transform(cv::Mat inputImage, cv::Mat &outputImage) = 0;
    /**
     * @brief Method to evaluate required characteristics from the target image.
     * This could be stain matrix, mean, variances etc.,.
     * @param targetImage 
     */
    virtual void fit(cv::Mat targetImage) = 0;

protected:
    /**
     * @brief Construct a new Normalizer object, default contructor.
     * 
     */
    Normalizer() = default;
    /**
     * @brief Destroy the Normalizer object.
     * 
     */
    virtual ~Normalizer(){};
};
