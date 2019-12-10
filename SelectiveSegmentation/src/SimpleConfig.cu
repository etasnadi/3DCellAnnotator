#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <fstream>
#include <boost/log/trivial.hpp>

#include "macros.h"

#include "SimpleConfig.cuh"
#include "common.cuh"

using namespace std;

#include <iostream>
#include <sstream>
#include <locale>

/*

Force the locale to use the C locale
@author: kriston

Example:
std::cout << from_string<float>(to_string(5.2)) << std::endl;

*/

const std::locale clocale("C");

template<typename CharType = char>
std::basic_ostringstream<CharType> ostringstream_locale(const std::locale& pLocale = clocale)
{
	std::basic_ostringstream<CharType> oss;
	oss.imbue(pLocale);
	return oss;
}

template<typename InType>
std::string _to_string(const InType& pValue, const std::locale& pLocale = clocale)
{
	std::basic_ostringstream<char> oss = ostringstream_locale<char>(pLocale);
	oss << pValue;
	return oss.str();
}


template<typename OutType, typename CharType = char>
OutType from_string(const CharType* pStr, const std::locale& pLocale = clocale)
{
	if(pStr == nullptr) throw std::runtime_error("Empty input string");

	std::basic_istringstream<CharType> iss(pStr);
	iss.imbue(pLocale);
	OutType value;
	if(!(iss >> value)) throw std::invalid_argument("Cannot read value");
	return value;
}

template<typename OutType, typename CharType = char>
OutType from_string(const std::basic_string<CharType>& pStr, const std::locale& pLocale = clocale)
{
	std::basic_istringstream<CharType> iss(pStr);
	iss.imbue(pLocale);
	OutType value;
	if(!(iss >> value)) throw std::invalid_argument("Cannot read value");
	return value;
}

/*
std::vector<std::string> split_string(const std::string& s, char delimiter){
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

// Locale independent atof because it is not recommended to change the locale.
float li_atof(string& s, char delim){
        vector<string> parts = split_string(s, delim);
        float result = 0.0f;
        if(parts.size() < 1 || parts.size() > 2){
                cerr << "Can't parse " << s << " as float. Returning 0.0f " << endl;
                return result;
        }

        if(parts.size() > 0){
                result += stoi(parts[0]);
        }


        if(parts.size() > 1){
                int fpart = stoi(parts[1]);
                int fpart_ndigits = pow(10,parts[1].length());
                result +=  float(fpart)/float(fpart_ndigits);
        }

        return result;
}

// Replaces the locale dependent delimiter to ours
bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}
*/

SimpleConfig::SimpleConfig(map<string, string>& defaultSettings){
	settings = defaultSettings;
}


EXPORT_SHARED SimpleConfig::SimpleConfig(){
}

SimpleConfig::SimpleConfig(string default_settings){
	this->addFromString(default_settings);
}

void SimpleConfig::setSettings(map<string, string>& newSettings){
	settings = newSettings;
}

void parseConfigEntry(string entry, map<string, string>& settings){
	vector<string> setting = ::split(entry, '=');
	if(setting.size()>=2){
		if(setting[0][0] != '#'){
			settings[setting[0]] = setting[1];
		}
	}
}

void SimpleConfig::addFromString(string configString){
	BOOST_LOG_TRIVIAL(info) << "Adding settings from string: " << configString;
	vector<string> setting_vec = ::split(configString, ',');
	for(int i = 0; i < setting_vec.size(); i++){
		parseConfigEntry(setting_vec[i], settings);
	}
}

EXPORT_SHARED void SimpleConfig::addFromConfigFile(string configFilePath){
	BOOST_LOG_TRIVIAL(info) << "Reading settings from file: " << configFilePath;

	std::ifstream input(configFilePath);
	string line;
	while(getline(input, line)){
		parseConfigEntry(line, settings);
	}
}

void SimpleConfig::addFromOther(SimpleConfig& o){
	/*
	BOOST_LOG_TRIVIAL(warning) << "Overwriting settings!";
	for (auto const& setting : o.getSettings()){
		if(o.getSettings().count(setting.first) > 0){
			BOOST_LOG_TRIVIAL(info) << setting.first <<  "<-" << setting.second;
			settings[setting.first] = setting.second;
		}
	}
	*/
}

void SimpleConfig::testProperty(string propName){
	if(settings.count(propName) < 1){
		BOOST_LOG_TRIVIAL(warning) << "Configuration property '" <<  propName << "' can not be found!";
	}
}

void SimpleConfig::setProperty(string propName, string propValue){
	settings[propName] = propValue;
}

void SimpleConfig::setFProperty(string propName, float propValue){
	string s_propValue = _to_string(propValue);
	std::cout << "Set " << s_propValue << "/" << propValue << std::endl;
	settings[propName] = s_propValue;
}

bool SimpleConfig::isSetProperty(string propName){
	return settings.count(propName) > 0;
}

string SimpleConfig::getProperty(string propName){
	testProperty(propName);
	return settings[propName];
}

EXPORT_SHARED float SimpleConfig::getFProperty(string propName){
	testProperty(propName);
	float result = from_string<float>(settings[propName]);
	BOOST_LOG_TRIVIAL(trace) << propName << "<-" <<  result;
	return result;
}

string SimpleConfig::getSProperty(string propName){
	testProperty(propName);
	string result = settings[propName];
	BOOST_LOG_TRIVIAL(trace) << propName << "<-" <<  result;
	return result;
}

EXPORT_SHARED int SimpleConfig::getIProperty(string propName){
	testProperty(propName);
	int result = atoi(settings[propName].c_str());

	BOOST_LOG_TRIVIAL(trace) << propName << "<-" <<  result;
	return result;
}

EXPORT_SHARED string& SimpleConfig::operator[] (const string propName){
	return settings[propName];
}

EXPORT_SHARED const string& SimpleConfig::operator[] (const string propName) const {
	return settings.at(propName);
}

void SimpleConfig::printSettings(){
	BOOST_LOG_TRIVIAL(info) << "Current configuration: ";
	for (auto const& setting : settings){
		BOOST_LOG_TRIVIAL(info) << setting.first
				  << '='
				  << setting.second
				  << std::endl ;
	}
	BOOST_LOG_TRIVIAL(info) << "=======================";
}

void SimpleConfig::generateConfig(std::ostream& stream){
	BOOST_LOG_TRIVIAL(info) << "Saving configuration...";
	std::stringstream buffer;
	std::time_t t = std::time(nullptr);
	std::tm tm = *std::localtime(&t);
	buffer << std::put_time(&tm, "%a_%d-%b-%Y_%H.%M.%S");

	stream << "# Current launch configuration: " << buffer.str() << std::endl;
	for (auto const& setting : settings){
		if(settings.count(setting.first) > 0){
			 stream << setting.first
					  << '='
					  << setting.second
					  << std::endl ;
		}else{
			BOOST_LOG_TRIVIAL(info) << "Omitting property '" << setting.first << "' because its value is empty!";
		}
	}
}


char* getArgValue(int argid, int argc, char** argv){
	int nextargid = argid+1;
	if(nextargid < argc){
		return argv[nextargid];
	}else{
		BOOST_LOG_TRIVIAL(fatal) << "No value set for argument!";
		throw 20;
	}
}

// Utility function to get a config from command line
SimpleConfig getConfig(int argc, char** argv){
	SimpleConfig conf;
	string settingsFname = SETTINGS_FILE_DEFAULT;

	if(argc == 1){
		BOOST_LOG_TRIVIAL(info) << "Using a default config file: " << settingsFname;
		conf.addFromConfigFile(SETTINGS_FILE_DEFAULT);
	}

	for(int i = 1; i < argc; i += 2){
		char* argname = argv[i];
		char* argval = getArgValue(i, argc, argv);
		if(strcmp(argname, "--add-settings-file") == 0){
			conf.addFromConfigFile(argval);
		}else if(strcmp(argname, "--add-config") == 0){
			conf.addFromString(argval);
		}else{
			BOOST_LOG_TRIVIAL(warning) << "Can't parse argument " << i << "(" << argv[i] << ")";
		}
	}

	return conf;
}

