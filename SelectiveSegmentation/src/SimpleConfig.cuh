/*
 * SimpleConfig.cuh
 *
 *  Created on: 2018 aug. 29
 *      Author: ervin
 */

#ifndef SIMPLECONFIG_CUH_
#define SIMPLECONFIG_CUH_

#include "macros.h"

#include <string>
#include <map>
#include <vector>
using namespace std;

// Config class

#define FLOAT_DELIM '.'
#define FLOAT_DELIM_S "."
#define SETTINGS_FILE_DEFAULT "settings.conf"

class SimpleConfig {
private:
	map<string, string> settings;
	vector<string> split(const string &s, char delim);
	void testProperty(string propName);
public:
	EXPORT_SHARED SimpleConfig();
	SimpleConfig(map<string, string>& default_settings);
	SimpleConfig(string default_settings);
	map<string, string> getSettings();
	void setSettings(map<string, string>& settings);
	void addFromString(string argument);
	EXPORT_SHARED void addFromConfigFile(string file_path);
	void addFromOther(SimpleConfig& o);
	string getProperty(string propName);
	void setProperty(string propName, string value);
	void setFProperty(string propName, float propValue);
	bool isSetProperty(string propName);
	EXPORT_SHARED float getFProperty(string propName);
	string getSProperty(string propName);
	EXPORT_SHARED int getIProperty(string propName);
	EXPORT_SHARED string& operator[] (const string propName);
	EXPORT_SHARED const string& operator[] (const string propName) const;
    void printSettings();
    void generateConfig(std::ostream& stream);
};

SimpleConfig getConfig(int argc, char** argv);

#endif /* SIMPLECONFIG_CUH_ */
