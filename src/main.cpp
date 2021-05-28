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
using namespace std;
using namespace ColourStainNormalization;
#if __cplusplus == 201703L
#include <filesystem>
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
struct DataUnit
{
    public:
    cv::Mat inputImage;
    std::string fileName;
    cv::Mat outputImage;
    ~DataUnit(){
        inputImage.release();
        outputImage.release();
    }
};
static std::optional<DataUnit> test() { return std::nullopt; }
int main(int argc, char *argv[])
{
    std::unique_ptr<Normalizer> p_normalizer = VahadaneNormalizer::New();
    std::string dirName = "../src/";
    std::string outputDir = "./";
    std::string referenceImageName;
    std::string mode = "v";
    std::string file_list;
    std::stringstream strstream;
    std::string outputdir;
    vector<string> fileList;
    bool invalidMode = false;
    bool helpMode = false;
    
    if (argc < 3)
    {
        exit_with_help();
        return EXIT_SUCCESS;
    }
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
            cout << "Target"
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
                p_normalizer = MacenkoNormalizer::New();
                outputDir+="_Macenko";
                break;
            }
            if (tolower(argv[i][0]) == 'v')
            {
                p_normalizer = VahadaneNormalizer::New();
                outputDir+="_Vahadane";
                break;
            }
            if (tolower(argv[i][0]) == 'r')
            {
                p_normalizer = ReinhardNormalizer::New();
                outputDir+="_Reinhard";
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
    if (invalidMode)
    {
        cerr << "Invalid Normalization Method specified"
             << "\n";
        cerr << "Valid Choices are m: Macenko, v: Vahadane, r: Reinhard"
             << "\n";
        return EXIT_FAILURE;
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
    
    parseList(dirName, file_list, fileList);

    cv::Mat image = cv::imread(referenceImageName); //
    int size = fileList.size();
    std::vector<DataUnit> imageList;
    for(int i = 0; i < size; i++)
    {
        DataUnit du;
        du.fileName = fileList.at(i);
        du.inputImage = cv::imread(dirName + "/" + (du.fileName));
        imageList.push_back(du);
    }
    auto start = chrono::steady_clock::now();
    p_normalizer->fit(image);
    #pragma omp parallel num_threads(NUM_THREADS << 1)
    {
        #pragma omp for
        for(int i = 0; i < size; i++)
        {
            p_normalizer->transform(imageList.at(i).inputImage, imageList.at(i).outputImage);
        }
    }
    auto end = chrono::steady_clock::now();
    cout << "Total Time Taken:(ms) ";
    cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
    cout << "Average Time Taken:(ms) ";
    cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() / fileList.size() << endl;
    for(int i = 0; i < size; i++){
        DataUnit du = imageList.at(i);
        imwrite(outputDir + "/" + (du.fileName), du.outputImage);
    }
    return EXIT_SUCCESS;
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
