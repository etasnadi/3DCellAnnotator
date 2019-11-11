#pragma once

#include <memory>

#include "ccalg/computeComponents.cuh"
#include "common.cuh"

typedef std::pair<unique_ptr<DevIntChk>, unique_ptr<DevFloatChk> > CompStats;
typedef std::pair<unique_ptr<DevIntChk>, unique_ptr<DeviceChunk<float4> > > CompStatsF4;
typedef std::pair<unique_ptr<DevIntChk>, unique_ptr<DeviceChunk<float3> > > CompStatsF3;

class CurveProps {
public:
	// Total:
	float surf;
	float vol;
	float3 cog;
	float sm;

	CompStats surfContribs;
	CompStats volContribs;
	CompStatsF3 cogContribs;
	CompStats smContribs;
};

void printStats(CurveProps& curParams, int nComps, ostream& stream);
CurveProps computeStats(unique_ptr<DevFloatChk>& ls1, unique_ptr<DevIntChk>& ccresult, int nComps, GridParams gParams);

float getVolume_global(DevFloatChk& data, DevFloatChk& volume, float dh, Size3D size);
float getSurface_global(DevFloatChk& data, DevFloatChk& surface, float dh, Size3D size);
