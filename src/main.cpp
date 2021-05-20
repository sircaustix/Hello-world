/**
 * @file main.cpp
 * @author Anand A (aanand@nference.net)
 * @brief The test class
 * @version 0.1
 * @date 2021-05-12
 * 
 * @copyright Copyright (c) 2021 nFerence Labs India Private Limited.
 * 
 */
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <MacenkoNormalizer.h>
#include <VahadaneNormalizer.h>
#include <ReinhardNormalizer.h>
#include <Utils.h>
#include <chrono>
#include <omp.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <thread>
using namespace std;
using namespace ColourStainNormalization;
#if __cplusplus == 201703L
namespace fs = std::filesystem;
#endif
/**
 * @brief Method to get the list of files in a directory.
 * @param dirName The directory name.
 * @param fileList std::vector<string> object containing the file-names.
 */
void getFilesList(std::string dirName, std::vector<string> &fileList);
/**
 * @brief 
 * @param dirName 
 * @param list 
 * @param fileList 
 */
void parseList(const string dirName, const string list, vector<string> &fileList);
void exit_with_help()
{
    cerr << "Correct Usage of the application:" << endl;
    cerr << "./main -r Reference Image Path " << endl;
    cerr << "-d directory containing the images to be normalized" << endl;
    cerr << "-o output directory to write the normalized images" << endl;
    cerr << "-l all to convert all files or comma separated list of files like img1.jpg, img2.jpg" << endl;
    cerr << "-m normalization mode, m(Macenko), v(Vahadane) or r(Reinhard)" << endl;
}
const int VAHADANE = 0;
const int MACENKO = 1;
const int REINHARD = 2;
const int NUM_THREADS = std::thread::hardware_concurrency();
int main(int argc, char *argv[])
{
    //std::unique_ptr<Normalizer> _normalizer = static_cast<std::unique_ptr<Normalizer>>(MacenkoNormalizer::New().get());
    //Normalizer &_normalizer;
    std::string dirName = "../src/";
    std::string outputDir = "./";
    std::string referenceImageName;
    std::string mode = "v";
    std::string file_list;
    std::stringstream strstream;
    std::string outputdir;
    vector<string> fileList;
    bool invalidMode = false;
    int stainMode = VAHADANE;
    bool helpMode = false;
    if (argc < 3)
    {
        exit_with_help();
        return EXIT_SUCCESS;
    }
    //cout << "The number of arguments = " << argc << "\n";
    //for(int i = 0; i < argc; i++)
    //    cout << argv[i] << "\n";
    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] != '-')
            break;
        if (++i >= argc)
        {
            exit_with_help();
            helpMode = true;
        }
        switch (tolower(argv[i - 1][1]))
        {
        case 'h':
            exit_with_help();
            helpMode = true;
            break;
        case 'r':
            referenceImageName = argv[i];
            cout << "Test"
                 << " " << referenceImageName << endl;
            break;
        case 'd':
            dirName = argv[i];
            cout << "Input Directory: " << dirName << endl;
            break;
        case 'o':
            outputDir = argv[i];
            cout << "Output Directory:"
                 << " " << outputDir << endl;
            break;
        case 'l':
            cout << "List: " << argv[i] << endl;
            file_list = argv[i];
            break;
        case 'm':
            mode = argv[i][0];
            cout << "Mode:" << mode << endl;
            if (tolower(argv[i][0]) == 'm')
            {
                stainMode = MACENKO;
                break;
            }
            if (tolower(argv[i][0]) == 'v')
            {
                stainMode = VAHADANE;
                break;
            }
            if (tolower(argv[i][0]) == 'r')
            {
                stainMode = REINHARD;
                break;
            }
            invalidMode = true;
            break;
        default:
            break;
        }
    }
    if (helpMode)
    {
        return EXIT_SUCCESS;
    }
    DIR *dir;
    if ((dir = opendir(dirName.c_str())) == NULL)
    {
        cerr << "The specified Read Directory Could not be opened\n";
        return EXIT_FAILURE;
    }
    if ((dir = opendir(outputDir.c_str())) == NULL)
    {
        if (mkdir(outputDir.c_str(), 0777))
        {
            cerr << "Unable To Create Output Dir" << endl;
            return EXIT_FAILURE;
        }
    }
    if (invalidMode)
    {
        cerr << "Invalid Normalization Method specified"
             << "\n";
        cerr << "Valid Choices are m: Macenko, v: Vahadane, r: Reinhard"
             << "\n";
        return EXIT_FAILURE;
    }
    parseList(dirName, file_list, fileList);

    cv::Mat image = cv::imread(referenceImageName); //
    auto start = chrono::steady_clock::now();

    if (VAHADANE == stainMode)
    {
        cerr << "Vahadane" << endl;
        VahadaneNormalizer normalizer = VahadaneNormalizer::init();
        normalizer.fit(image);
#pragma omp parallel num_threads(NUM_THREADS << 1)
        {
#pragma omp for
            for (vector<string>::iterator itr = fileList.begin(); itr != fileList.end(); ++itr)
            {
                // cout << dirName+"/"+(*itr) << "\n";
                cv::Mat image2 = cv::imread(dirName + "/" + (*itr));
                cv::Mat output;
                normalizer.transform(image2, output);
                imwrite(outputDir + "/" + (*itr), output);
                image2.release();
                output.release();
            }
        }
        auto end = chrono::steady_clock::now();
        cout << "Total Time Taken:(ms) ";
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
        cout << "Average Time Taken:(ms) ";
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() / fileList.size() << endl;
        return EXIT_SUCCESS;
    }
    if (MACENKO == stainMode)
    {
        cerr << "Macenko" << endl;
        MacenkoNormalizer normalizer = MacenkoNormalizer::init();
        normalizer.fit(image);

#pragma omp parallel num_threads(NUM_THREADS << 1)
#pragma omp for
        for (vector<string>::iterator itr = fileList.begin(); itr != fileList.end(); ++itr)
        {
            cout << dirName + "/" + (*itr) << "\n";
            cv::Mat image2 = cv::imread(dirName + "/" + (*itr));
            cv::Mat output;
            normalizer.transform(image2, output);
            imwrite(outputDir + "/" + (*itr), output);
            image2.release();
            output.release();
        }
        auto end = chrono::steady_clock::now();
        cout << "Total Time Taken:(ms) ";
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
        cout << "Average Time Taken:(ms) ";
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() / fileList.size() << endl;
        return EXIT_SUCCESS;
    }
    if (REINHARD == stainMode)
    {
        cerr << "Reinhard" << endl;
        ReinhardNormalizer normalizer = ReinhardNormalizer::init();
        normalizer.fit(image);
#pragma omp parallel num_threads(NUM_THREADS << 1)
#pragma omp for
        for (vector<string>::iterator itr = fileList.begin(); itr != fileList.end(); ++itr)
        {
            cout << dirName + "/" + (*itr) << "\n";
            cv::Mat image2 = cv::imread(dirName + "/" + (*itr));
            cv::Mat output;
            normalizer.transform(image2, output);
            imwrite(outputDir + "/" + (*itr), output);
            image2.release();
            output.release();
        }
        auto end = chrono::steady_clock::now();
        // cout<<"Total Time Taken:(ms) ";
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
        cout << "Average Time Taken:(ms) ";
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() / fileList.size() << endl;
        return EXIT_SUCCESS;
    }
    return 0;
}
void getFilesList(std::string dirName, std::vector<string> &fileList)
{
    fileList.clear();
#if __cplusplus == 201703L
    if (fs::exists(dirName))
    {

        for (const auto &entry : std::filesystem::directory_iterator(dirName))
        {
            size_t found = 0;
            if (!fs::is_directory(entry.path()))
            {
                string str = entry.path();
                found = str.find_last_of("/", str.length());
                //cout << str.substr(found+1, str.length()) << " " << found << endl;
                if (found != string::npos)
                {
                    fileList.push_back(str.substr(found + 1, str.length()));
                }
            }
        }
    }
    return;
#endif
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(dirName.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
#ifdef _DIRENT_HAVE_D_TYPE
            if (ent->d_type != DT_UNKNOWN && ent->d_type != DT_LNK && ent->d_type != DT_DIR)
            {
                fileList.push_back(ent->d_name);
            }
#else
            {
                struct stat statbuf;
                stat(ent->d_name, &statbuf);
                bool isDir = S_ISDIR(statbuf.st_mode);
                if (!isDir)
                {
                    fileList.push_back(ent->d_name);
                }
            }
#endif
        }
    }
}

void parseList(const string dirName, const string file_list, vector<string> &fileList)
{
    fileList.clear();
    stringstream stream(file_list);
    string intermediate;
    string original;
    while (getline(stream, intermediate, ','))
    {
        intermediate.erase(std::remove_if(intermediate.begin(), intermediate.end(), ::isspace), intermediate.end());
        //cout << intermediate << endl;
        original = intermediate;
        transform(intermediate.begin(), intermediate.end(), intermediate.begin(), ::tolower);
        if (intermediate.compare("all") == 0)
        {
            fileList.clear();
            getFilesList(dirName, fileList);
            cerr << "It was All" << endl;
            return;
        }
        else
        {
            fileList.push_back(original);
        }
    }
}