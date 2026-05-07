#ifndef UTILS_H
#define UTILS_H
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

typedef uint32_t uintPE;
typedef uint32_t uint32;
typedef uint16_t uint16;
typedef uint64_t uint64;

#define MAX_NUM 0xffffffff
#define UP_HALF 0xffff0000
#define DW_HALF 0x0000ffff
#define HF_OFST 16
#define MAX_BIT 0x80000000
#define MIN_BIT 0x7fffffff
#define SEC_BIT 0x40000000
#define THD_BIT 0x20000000

#define MIN_BIT 0x7fffffff
#define MIN_HFUP 0x7fff0000
#define MAX_OFS 31
#define SEC_OFS 30
#define THD_OFS 29
#define DW_18bit 0x0003ffff

#include <omp.h>
#define cilk_spawn
#define cilk_sync
#define parallel_main main
#define parallel_for _Pragma("omp parallel for") for
#define parallel_for_1 _Pragma("omp parallel for schedule (static,1)") for
#define parallel_for_256 _Pragma("omp parallel for schedule (static,256)") for

#define newA(__E, __n) (__E *)malloc((__n) * sizeof(__E))

struct words {
  char *Chars;     // array storing all strings
  long n;          // total number of characters
  char **Strings;  // pointers to strings (all should be null terminated)
  long m;          // number of substrings
  words() {}

  words(char *C, long nn, char **S, long mm)
      : Chars(C), n(nn), Strings(S), m(mm) {}

  void del() {
    free(Chars);
    free(Strings);
  }
};

struct commandLine {
  int argc;
  char **argv;
  std::string comLine;

  commandLine() {}

  commandLine(int _c, char **_v, std::string _cl)
      : argc(_c), argv(_v), comLine(_cl) {
    if (getOption("-h") || getOption("-help")) badArgument();
  }

  commandLine(int _c, char **_v)
      : argc(_c), argv(_v), comLine("bad arguments") {}

  void badArgument() {
    std::cout << "usage: " << argv[0] << " " << comLine << std::endl;
    exit(0);
  }

  char *getArgument(int i) {
    if (argc < 2 + i) badArgument();
    return argv[argc - 1 - i];
  }

  std::pair<char *, char *> IOFileNames() {
    if (argc < 3) badArgument();
    return std::pair<char *, char *>(argv[argc - 2], argv[argc - 1]);
  }

  std::pair<size_t, char *> sizeAndFileName() {
    if (argc < 3) badArgument();
    return std::pair<size_t, char *>(
        std::atoi(argv[argc - 2]), (char *)argv[argc - 1]);
  }

  bool getOption(std::string option) {
    for (int i = 1; i < argc; i++)
      if ((std::string)argv[i] == option) return true;
    return false;
  }

  char *getOptionValue(std::string option) {
    for (int i = 1; i < argc - 1; i++)
      if ((std::string)argv[i] == option) return argv[i + 1];
    return NULL;
  }

  std::string getOptionValue(std::string option, std::string defaultValue) {
    for (int i = 1; i < argc - 1; i++)
      if ((std::string)argv[i] == option) return (std::string)argv[i + 1];
    return defaultValue;
  }

  long getOptionLongValue(std::string option, long defaultValue) {
    for (int i = 1; i < argc - 1; i++)
      if ((std::string)argv[i] == option) {
        long r = atol(argv[i + 1]);
        if (r < 0) badArgument();
        return r;
      }
    return defaultValue;
  }

  int getOptionIntValue(std::string option, int defaultValue) {
    for (int i = 1; i < argc - 1; i++)
      if ((std::string)argv[i] == option) {
        int r = atoi(argv[i + 1]);
        if (r < 0) badArgument();
        return r;
      }
    return defaultValue;
  }

  double getOptionDoubleValue(std::string option, double defaultValue) {
    for (int i = 1; i < argc - 1; i++)
      if ((std::string)argv[i] == option) {
        double val;
        if (sscanf(argv[i + 1], "%lf", &val) == EOF) {
          badArgument();
        }
        return val;
      }
    return defaultValue;
  }
};

class Logger {
 public:
  Logger(const std::string &filePath) : logFile(filePath, std::ios::app) {
    if (!logFile.is_open()) {
      throw std::runtime_error("Failed to open log file: " + filePath);
    }
  }

  ~Logger() {
    if (logFile.is_open()) {
      logFile.close();
    }
  }

  template <typename T>
  Logger &operator<<(const T &value) {
    if (!firstColumn) {
      logFile << ',';
    }
    logFile << value;
    firstColumn = false;
    return *this;
  }

  Logger &end() {
    logFile << std::endl;
    firstColumn = true;
    return *this;
  }

 private:
  std::ofstream logFile;
  bool firstColumn = true;
};

inline bool isSpace(char c);

unsigned int getBinaryLength(unsigned int x);

words stringToWords(char *Str, uint64_t n);

char *readStringFromFile(const char *fileName, long *length);

inline void set_num_workers(int n) { omp_set_num_threads(n); }

static int getWorkers() { return omp_get_max_threads(); }

#endif  // UTILS_H
