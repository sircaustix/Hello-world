#pragma once
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <thread>
/**
 * @brief The constant integer variable that contains the number of processors/cores
 * present in the machine.
 * 
 */
const unsigned int NUM_THREADS = std::thread::hardware_concurrency();
/**
 * @brief Namespace ColourStainNormalization, contains forward declaration of all the classes
 * that are part of the namespace.
 * 
 */
namespace ColourStainNormalization
{
    class Utils;
    class Normalizer;
    class MacenkoNormalizer;
    class MatrixMethods;
    class ReinhardNormalizer;
    class VahadaneNormalizer;
};