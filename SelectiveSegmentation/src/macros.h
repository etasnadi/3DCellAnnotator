#ifndef MACROS_H
#define MACROS_H

#ifndef EXPORT_SHARED
#define EXPORT_SHARED
#endif

#ifdef _WIN32
#ifdef EXPORT_SHARED
#undef EXPORT_SHARED
#endif

#define EXPORT_SHARED __declspec(dllexport)
#endif

#endif