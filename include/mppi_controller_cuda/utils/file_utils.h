#ifndef FILE_UTILS_H_
#define FILE_UTILS_H_

#include <string>
#include <unistd.h>

inline bool fileExists(const std::string& name)
{
  return (access(name.c_str(), F_OK) != -1);
}

#endif  // FILE_UTILS_H_
