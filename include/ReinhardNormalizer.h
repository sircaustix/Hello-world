/**
 * @file ReinhardNormalizer.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-05-29
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#pragma once
#include "Normalizer.h"
/**
 * @brief Implementation of Reinhard Normalizer. 
 * Derived from Normalizer class.
 */
class ColourStainNormalization::ReinhardNormalizer : public ColourStainNormalization::Normalizer
{
    //friend std::default_delete<ColourStainNormalization::ReinhardNormalizer>;

public:
    /**
     * @brief Static Method to create a unique_ptr to the object.
     * 
     * @return std::unique_ptr<ColourStainNormalization::Normalizer> 
     */
    static std::unique_ptr<ColourStainNormalization::Normalizer> New();
    /**
     * @brief Method to instantiate the object.
     * 
     * @return ColourStainNormalization::ReinhardNormalizer 
     */
    static ColourStainNormalization::ReinhardNormalizer init() { return ReinhardNormalizer(); }
    /**
     * @brief Method to transform the colours of an input image to that of
     * a target image.
     * @param source , the input image
     * @param output , the transformed image.
     */
    void transform(cv::Mat source, cv::Mat &output);
    /**
     * @brief Method to evaluate required characteristics from the target image.
     * In this case the mean and standard deviation.
     * 
     * @param target 
     */
    void fit(cv::Mat target);

protected:
    /**
     * @brief Construct a new Reinhard Normalizer object
     * 
     */
    ReinhardNormalizer() = default;

private:
    /**
     * @brief Method to convert an input image in LAB space in to individual
     * L, A and B channels.
     * @param I , the input image.
     * @param I1 , the L channel.
     * @param I2 , the A channel.
     * @param I3 , the B channel.
     */
    void labSplit(cv::Mat I, cv::Mat &I1, cv::Mat &I2, cv::Mat &I3);
    /**
     * @brief Method to merge individual L, A and B channels to an LAB image.
     * 
     * @param I1 , L Channel. 
     * @param I2 , A Channel.
     * @param I3 , B Channel.
     * @return cv::Mat the merged output image.
     */
    cv::Mat mergeBack(cv::Mat I1, cv::Mat I2, cv::Mat I3);
    /**
     * @brief Get the Mean and std of the input image. 
     * 
     * @param I, the input image. 
     * @param mean, the mean of the image.
     * @param std, the standard deviation of the image. 
     */
    void getMeanStd(cv::Mat I, cv::Scalar &mean, cv::Scalar &std);
    /**
     * @brief The variable to store the mean of the target image.
     * This is computed by invoking the fit method.
     */
    cv::Scalar targetMean;
    /**
     * @brief The variable to store the std of the target image.
     * This is computed in the fit method.
     */
    cv::Scalar targetStd;
};