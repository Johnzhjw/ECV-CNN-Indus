#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "MOP_IWSN.h"

#define resIWSN (2)
#define lon ((30+1)/resIWSN+1)
#define wid ((37+1)/resIWSN+1)
#define hig ((25+1)/resIWSN+1)
#define hig0 (1)
#define hig1 (5)
#define hig2 (9)
#define hig3 (14)
#define D1 (0)
#define D2 (1)
#define D3 (2)
#define X (0)
#define Y (1)
#define Z (2)
#define PAN (3)
#define TILT (4)
#define PI (3.14)
#define pi (3.1415926)
#define alpha1_direc (1.0)
#define alpha2_direc (0.0)
#define beta1_direc (1.5)
#define beta2_direc (1.0)
#define r_direc_min (45.0*2.0/3.0/resIWSN)
#define r_direc_max (72.0*2.0/3.0/resIWSN)
#define r_direc_ratio (1.0)
#define N ((lon)*(wid)*(hig))//
#define ga (-0.5)
#define oq_beta (0.9)
#define pan_min (pi/4*2/3/2)
#define pan_max (pi/3*2/3/2)
#define tilt_min (pi/4*2/3/2)
#define tilt_max (pi/3*2/3/2)
#define penaltyVal (1e6)
#define tau1_pan (3.6)
#define tau2_pan (3.6)
#define vu_pan (0.5)
#define mu_pan (0.5)
#define tau1_tilt (3.6)
#define tau2_tilt (3.6)
#define vu_tilt (0.5)
#define mu_tilt (0.5)
#define pan_ratio_mid (1.5)
#define pan_ratio_upp (2.0)
#define tilt_ratio_mid (1.5)
#define tilt_ratio_upp (2.0)

#define LWratio (1.33333333)

#define angE (2.0 * pi)

#define angEmin (-pi/2.0)
#define angEmax (pi/2.0)

#define minLIM (0.0)
#define maxLIM (1.0-1e-9)

static int map[lon][wid][hig] = {
	{
		//0
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//1
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//2
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//3
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//4
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//5
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//6
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//7
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//8
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//9
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//10
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//11
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//12
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//13
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//14
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	},
	{
		//15
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
	}
};

static double qoc[lon][wid][hig];
static double radiusRs_DIREC[N_DIREC];
static double radiusRf_DIREC[N_DIREC];
static double pan_angle[N_DIREC];
static double tilt_angle[N_DIREC];
static double pan_range[N_DIREC];
static double tilt_range[N_DIREC];

static double occupVol = 0.0; //volume occupied by devices
// static int nPenalty;
static int pos3D_DIREC[N_DIREC][3];
static int pos3D_RELAY[N_RELAY][3];
static int pos3D_SENSOR[N_SENSOR][3];
static bool posFlag_DIREC[N_DIREC];
static bool posFlag_RELAY[N_RELAY];
static bool posFlag_SENSOR[N_SENSOR];
static bool posCEIL_SENSOR[N_SENSOR];
static int posCountBad;

//lifetime
#define energy_ini (10.0)
#define E_elec (50.0e-9)
#define e_fs (10.0e-12)
#define e_mp (0.0013e-12)
#define d_th (87.0)
#define E_DA (5.0e-9)
#define d_DA (0.1)
#define l0 (200)
#define n_rn_min (2)
#define d_th_sn (1.0/8.0*d_th)
#define d_th_rn (1.0/4.0*d_th)

// position of the sink node
#define SINK_X (lon/2)
#define SINK_Y (wid-1)
#define SINK_Z (hig/2)

static int hopID_SENSOR[N_SENSOR];  // 0,1,2,...,N_RELAY-1, N_RELAY - SINK
static int hopID_RELAY[N_RELAY];   // 0,1,2,...,N_RELAY-1, N_RELAY - SINK
static double com_dist_SENSOR[N_SENSOR];
static double com_dist_RELAY[N_RELAY];
static double n_data_local_RELAY[N_RELAY];
static double n_data_hop_RELAY[N_RELAY];
static double n_data_fwd_RELAY[N_RELAY];

double avg_dist_SENSOR_IWSN;
// static double std_dist_SENSOR;
static double energy_consumed_RELAY[N_RELAY];
static double LT_RELAY[N_RELAY];
static double LT_avg;
double LT_std_IWSN;
double LT_min_IWSN;

static int n_sn_rn_com[N_SENSOR];
static int n_rn_rn_com[N_RELAY];
static int n_relia_p = 0;

static double distNode2Relay[N_SENSOR + N_RELAY][N_RELAY];

/***************************************************/
//
// static int RandomInteger(int low, int high);
static double RandomDouble(double low, double high);
static bool TransPos_2D3D(int bD, double vTilt, double r1, double r2, double r3, int* _3D, bool* _posCEIL);
static double range(int i, int j, int k, int a, int b, int c); //
static int LOS(int i, int j, int h, int a, int b, int c);
static double Oq_DIREC(int i, int j, int k, int l); //
static void Qoc();//
static double Cover();//
static double Lifetime();//
static double Reliability();//
// static double Control_Angle(double a);//
static double ArcPan(int i, int j, int k, int a, int b, int c); //
static double ArcTilt(int i, int j, int k, int a, int b, int c); //tilt angle
// static double Myerf(double a);
/***************************************************/
/***************************************************/
/*
int main()
{
occupVol = 0.0;

int i, j, k;
for (i = 0; i < lon; i++) {
for (j = 0; j < wid; j++) {
for (k = 0; k < hig; k++) {
if (map[i][j][k])
occupVol += 1.0;
}
}
}
//occupVol /= N;
srand(666);
double individual[40][DIM_IWSN];
double fitness[IWSNOBJ];

for (k = 0; k < N_DIREC; k++) {
radiusRs_DIREC[k] = RandomDouble(r_direc_min, r_direc_max);
radiusRf_DIREC[k] = radiusRs_DIREC[k] * r_direc_ratio;
pan_range[k] = RandomDouble(pan_min, pan_max);
tilt_range[k] = pan_range[k];
}

for (i = 0; i < 40; i++) {
for (int a = 0; a < DIM_IWSN; a++) {
individual[i][a] = minLIM + 0.5 * (maxLIM - minLIM);
}
if (i % 2 == 0)
individual[i][107] =
((int)((i) / 2) + 0.5) / 20 * (maxLIM - minLIM) + minLIM;
else
individual[i][110] =
((int)((i) / 2) + 0.5) / 20 * (maxLIM - minLIM) + minLIM;
Fitness_IWSN(&individual[i][0], fitness);
for (int a = 0; a < DIM_IWSN; a++) {
printf("%.16lf ", individual[i][a]);
}
printf("\n");
printf("coverage:%.16lf, %d %d\n", fitness[0], posCountBad, n_relia_p);
printf("lifetime:%.16lf\n", fitness[1]);
printf("reliability:%.16lf\n", fitness[2]);
}

for (i = 0; i < 1000; i++) {
for (k = 0; k < DIM_IWSN; k++) {
individual[0][k] = RandomDouble(minLIM, maxLIM);
}
Fitness_IWSN(individual[0], fitness);
printf("coverage:%.16lf, %d %d\n", fitness[0], posCountBad, n_relia_p);
printf("lifetime:%.16lf\n", fitness[1]);
printf("reliability:%.16lf\n", fitness[2]);
}

return 0;
}
*/

void SetLimits_IWSN(double* minLimit, double* maxLimit, int nx)
{
	srand(666);

	int i, j, k;
	for (k = 0; k < N_DIREC; k++) {
		radiusRs_DIREC[k] = RandomDouble(r_direc_min, r_direc_max);
		radiusRf_DIREC[k] = radiusRs_DIREC[k] * r_direc_ratio;
		pan_range[k] = RandomDouble(pan_min, pan_max);
		tilt_range[k] = pan_range[k];
	}

	for (k = 0; k < DIM_IWSN; k++) {
		minLimit[k] = minLIM;
		maxLimit[k] = maxLIM;
	}

	occupVol = 0.0;

	for (i = 0; i < lon; i++) {
		for (j = 0; j < wid; j++) {
			for (k = 0; k < hig; k++) {
				if (map[i][j][k])
					occupVol += 1.0;
			}
		}
	}
	//occupVol /= N;
}

int CheckLimits_IWSN(double* x, int nx)
{
	int k;

	for (k = 0; k < DIM_IWSN; k++) {
		if (x[k] < minLIM || x[k] > maxLIM) {
			printf("Check limits FAIL - IWSN: %d\n", k);
			return false;
		}
	}

	return true;
}

static bool TransPos_2D3D(int bD, double vTilt, double r1, double r2, double r3, int* _3D, bool* _posCEIL)
{
	int facet = (int)(r1 * 15);
	int pX, pY, pZ;
	bool rB = false;

	int theHeight1, theHeight2;

	if (facet / 5 == 0) {  //cabin 1
		theHeight1 = hig0;
		theHeight2 = hig1;
	}
	else if (facet / 5 == 1) {  //cabin 2
		theHeight1 = hig1;
		theHeight2 = hig2;
	}
	else { // cabin 3
		theHeight1 = hig2;
		theHeight2 = hig3;
	}

	if (facet % 5 == 0) {  //front
		pX = 1 + (int)((lon - 2) * r2);
		pY = 0;
		pZ = theHeight1 + (int)((theHeight2 - theHeight1 - 1) * r3);

		if (_3D) {
			_3D[X] = pX;
			_3D[Y] = pY + 1;
			_3D[Z] = pZ;
		}
		if ((map[pX][pY][pZ] == 1) && (map[pX][pY + 1][pZ] == 0)) {
			if (bD) {
				/*if (((!map[pX - 1][pY + 1][pZ]) && (!map[pX - 1][pY + 1][pZ + 1]) && (!map[pX][pY + 1][pZ + 1])) ||
				((!map[pX][pY + 1][pZ + 1]) && (!map[pX + 1][pY + 1][pZ + 1]) && (!map[pX + 1][pY + 1][pZ])) ||
				((!map[pX + 1][pY + 1][pZ]) && (!map[pX + 1][pY + 1][pZ - 1]) && (!map[pX][pY + 1][pZ - 1])) ||
				((!map[pX][pY + 1][pZ - 1]) && (!map[pX - 1][pY + 1][pZ - 1]) && (!map[pX - 1][pY + 1][pZ]))) {
				rB = true;
				}*/
				rB = true;
				_posCEIL[0] = 0;
			}
			else {
				rB = true;
			}
		}
	}
	else if (facet % 5 == 1) {  //right
		pX = 0;
		pY = 1 + (int)((wid - 2) * r2);
		pZ = theHeight1 + (int)((theHeight2 - theHeight1 - 1) * r3);

		if (_3D) {
			_3D[X] = pX + 1;
			_3D[Y] = pY;
			_3D[Z] = pZ;
		}
		if ((map[pX][pY][pZ] == 1) && (map[pX + 1][pY][pZ] == 0)) {
			if (bD) {
				/*if (((!map[pX + 1][pY - 1][pZ]) && (!map[pX + 1][pY - 1][pZ + 1]) && (!map[pX + 1][pY][pZ + 1])) ||
				((!map[pX + 1][pY][pZ + 1]) && (!map[pX + 1][pY + 1][pZ + 1]) && (!map[pX + 1][pY + 1][pZ])) ||
				((!map[pX + 1][pY + 1][pZ]) && (!map[pX + 1][pY + 1][pZ - 1]) && (!map[pX + 1][pY][pZ - 1])) ||
				((!map[pX + 1][pY][pZ - 1]) && (!map[pX + 1][pY - 1][pZ - 1]) && (!map[pX + 1][pY - 1][pZ]))) {
				rB = true;
				}*/
				rB = true;
				_posCEIL[0] = 0;
			}
			else {
				rB = true;
			}
		}
	}
	else if (facet % 5 == 2) {  //back
		pX = 1 + (int)((lon - 2) * r2);
		pY = wid - 1;
		pZ = theHeight1 + (int)((theHeight2 - theHeight1 - 1) * r3);

		if (_3D) {
			_3D[X] = pX;
			_3D[Y] = pY - 1;
			_3D[Z] = pZ;
		}
		if ((map[pX][pY][pZ] == 1) && (map[pX][pY - 1][pZ] == 0)) {
			if (bD) {
				/*if (((!map[pX - 1][pY - 1][pZ]) && (!map[pX - 1][pY - 1][pZ + 1]) && (!map[pX][pY - 1][pZ + 1])) ||
				((!map[pX][pY - 1][pZ + 1]) && (!map[pX + 1][pY - 1][pZ + 1]) && (!map[pX + 1][pY - 1][pZ])) ||
				((!map[pX + 1][pY - 1][pZ]) && (!map[pX + 1][pY - 1][pZ - 1]) && (!map[pX][pY - 1][pZ - 1])) ||
				((!map[pX][pY - 1][pZ - 1]) && (!map[pX - 1][pY - 1][pZ - 1]) && (!map[pX - 1][pY - 1][pZ]))) {
				rB = true;
				}*/
				rB = true;
				_posCEIL[0] = 0;
			}
			else {
				rB = true;
			}
		}
	}
	else if (facet % 5 == 3) {  //left
		pX = lon - 1;
		pY = 1 + (int)((wid - 2) * r2);
		pZ = theHeight1 + (int)((theHeight2 - theHeight1 - 1) * r3);

		if (_3D) {
			_3D[X] = pX - 1;
			_3D[Y] = pY;
			_3D[Z] = pZ;
		}
		if ((map[pX][pY][pZ] == 1) && (map[pX - 1][pY][pZ] == 0)) {
			if (bD) {
				/*if (((!map[pX - 1][pY - 1][pZ]) && (!map[pX - 1][pY - 1][pZ + 1]) && (!map[pX - 1][pY][pZ + 1])) ||
				((!map[pX - 1][pY][pZ + 1]) && (!map[pX - 1][pY + 1][pZ + 1]) && (!map[pX - 1][pY + 1][pZ])) ||
				((!map[pX - 1][pY + 1][pZ]) && (!map[pX - 1][pY + 1][pZ - 1]) && (!map[pX - 1][pY][pZ - 1])) ||
				((!map[pX - 1][pY][pZ - 1]) && (!map[pX - 1][pY - 1][pZ - 1]) && (!map[pX - 1][pY - 1][pZ]))) {
				rB = true;
				}*/
				rB = true;
				_posCEIL[0] = 0;
			}
			else {
				rB = true;
			}
		}
	}
	else if (facet % 5 == 4) {  //ceil
		pX = 1 + (int)((lon - 2) * r2);
		pY = 1 + (int)((wid - 2) * r3);
		pZ = theHeight2 - 1;

		if (_3D) {
			_3D[X] = pX;
			_3D[Y] = pY;
			_3D[Z] = pZ - 1;
		}
		if ((map[pX][pY][pZ] == 1) && (map[pX][pY][pZ - 1] == 0)) {
			if (bD) {
				/*if (((!map[pX - 1][pY][pZ - 1]) && (!map[pX - 1][pY + 1][pZ - 1]) && (!map[pX][pY + 1][pZ - 1])) ||
				((!map[pX][pY + 1][pZ - 1]) && (!map[pX + 1][pY + 1][pZ - 1]) && (!map[pX + 1][pY][pZ - 1])) ||
				((!map[pX + 1][pY][pZ - 1]) && (!map[pX + 1][pY - 1][pZ - 1]) && (!map[pX][pY - 1][pZ - 1])) ||
				((!map[pX][pY - 1][pZ - 1]) && (!map[pX - 1][pY - 1][pZ - 1]) && (!map[pX - 1][pY][pZ - 1]))) {
				rB = true;
				if (vTilt > 0)
				rB = false;
				}*/
				rB = true;
				_posCEIL[0] = 1;
			}
			else {
				rB = true;
			}
		}
	}

	return rB;
}

// static int RandomInteger(int low, int high)
// {
//     return (low + (int)(rand() / (RAND_MAX + 1.0) * (high - low + 1)));
// }

static double RandomDouble(double low, double high)
{
	return (low + (rand() / (RAND_MAX + 0.0)) * (high - low));
}

static double range(int i, int j, int k, int a, int b, int c) //
{
	double r;
	r = sqrt((double)((i - a) * (i - a) + (j - b) * (j - b) + (k - c) * (k - c)));
	return r;
}

static int LOS(int i, int j, int h, int a, int b, int c)                          ///LOS
{
	int m;
	int x, y, z;
	int x1, y1, z1;
	double k, k1;         //
	double x2;            //
	double y2;
	double z2;
	int x3, x4;
	int y3, y4;        //
	int z3, z4;
	x = abs(a - i);    //
	y = abs(b - j);
	z = abs(c - h);
	if (x >= y && x >= z && i != a) {
		x1 = (a - i) / x;         //
		k = (double)(b - j) / (double)(a - i);
		k1 = (double)(c - h) / (double)(a - i);
		for (m = 1; m < x; m++) {
			x2 = i + x1 * m;
			y2 = k * (x2 - i) + j;
			z2 = k1 * (x2 - i) + h; ////
			x3 = (int)x2;
			if (y2 == (int)y2) {
				y3 = (int)y2;
				if (z2 == (int)z2) {
					z3 = (int)z2;
					if (map[x3][y3][z3])
						return 0;
				}
				else {
					z3 = (int)z2;
					z4 = z3 + 1;
					if (map[x3][y3][z3] || map[x3][y3][z4])
						return 0;
				}
			}
			else {
				y3 = (int)y2;
				y4 = y3 + 1;
				if (z2 == (int)z2) {
					z3 = (int)z2;
					if (map[x3][y3][z3] || map[x3][y4][z3])
						return 0;
				}
				else {
					z3 = (int)z2;
					z4 = z3 + 1;
					if (map[x3][y3][z3] || map[x3][y3][z4] || map[x3][y4][z3] || map[x3][y4][z4])
						return 0;
				}
			}
		}
	}
	if (y >= x && y >= z && j != b) {
		y1 = (b - j) / y;
		k = (double)(a - i) / (double)(b - j); //
		k1 = (double)(c - h) / (double)(b - j);
		for (m = 1; m < y; m++) {
			y2 = j + y1 * m;
			x2 = k * (y2 - j) + i;
			z2 = k1 * (y2 - j) + h; ////
			y3 = (int)y2;
			if (x2 == (int)x2) {
				x3 = (int)x2;
				if (z2 == (int)z2) {
					z3 = (int)z2;
					if (map[x3][y3][z3])
						return 0;
				}
				else {
					z3 = (int)z2;
					z4 = z3 + 1;
					if (map[x3][y3][z3] || map[x3][y3][z4])
						return 0;
				}
			}
			else {
				x3 = (int)x2;
				x4 = x3 + 1;
				if (z2 == (int)z2) {
					z3 = (int)z2;
					if (map[x3][y3][z3] || map[x4][y3][z3])
						return 0;
				}
				else {
					z3 = (int)z2;
					z4 = z3 + 1;
					if (map[x3][y3][z3] || map[x3][y3][z4] || map[x4][y3][z3] || map[x4][y3][z4])
						return 0;
				}
			}
		}
	}
	if (z >= x && z >= y && h != c) {
		z1 = (c - h) / z;         //
		k = (double)(a - i) / (double)(c - h);
		k1 = (double)(b - j) / (double)(c - h);
		for (m = 1; m < z; m++) {
			z2 = h + z1 * m;
			x2 = k * (z2 - h) + i;
			y2 = k1 * (z2 - h) + j; ////
			z3 = (int)z2;
			if (x2 == (int)x2) {
				x3 = (int)x2;
				if (y2 == (int)y2) {
					y3 = (int)y2;
					if (map[x3][y3][z3])
						return 0;
				}
				else {
					y3 = (int)y2;
					y4 = y3 + 1;
					if (map[x3][y3][z3] || map[x3][y4][z3])
						return 0;
				}
			}
			else {
				x3 = (int)x2;
				x4 = x3 + 1;
				if (y2 == (int)y2) {
					y3 = (int)y2;
					if (map[x3][y3][z3] || map[x4][y3][z3])
						return 0;
				}
				else {
					y3 = (int)y2;
					y4 = y3 + 1;
					if (map[x3][y3][z3] || map[x3][y4][z3] || map[x4][y3][z3] || map[x4][y4][z3])
						return 0;
				}
			}
		}
	}

	return 1;
}

static double Oq_DIREC(int i, int j, int k, int l) //
{
	if (map[i][j][k])  //点在设备中
		return 0.0;
	if (pos3D_DIREC[l][X] == i && pos3D_DIREC[l][Y] == j && pos3D_DIREC[l][Z] == k)  //
		return 1.0;
	if (LOS(pos3D_DIREC[l][X], pos3D_DIREC[l][Y], pos3D_DIREC[l][Z], i, j, k)) {
		double r;        //
		r = range(pos3D_DIREC[l][X], pos3D_DIREC[l][Y], pos3D_DIREC[l][Z], i, j, k);
		double pan;
		if (pos3D_DIREC[l][X] == i && pos3D_DIREC[l][Y] == j)
			pan = 0;
		else
			pan = pan_angle[l] - ArcPan(pos3D_DIREC[l][X], pos3D_DIREC[l][Y], pos3D_DIREC[l][Z], i, j, k); //(-2pi,2pi)
		if (pan < 0)
			pan = -pan; //[0,2pi)
		if (pan > pi)
			pan = 2 * pi - pan; //an为[0,pi]，像素点相对传感器的偏转角
		//printf("%f\t",pan);
		double tilt;
		tilt = tilt_angle[l] - ArcTilt(pos3D_DIREC[l][X], pos3D_DIREC[l][Y], pos3D_DIREC[l][Z], i, j, k);
		if (tilt < 0)
			tilt = -tilt;
		if (tilt > pi)
			tilt = 2 * pi - tilt;
		tilt *= LWratio;
		//printf("%f\n",tilt);
		double prob_d, prob_pan, prob_tilt;
		if (r <= radiusRs_DIREC[l])
			prob_d = 1.0;
		else if (r > radiusRs_DIREC[l] && r < radiusRs_DIREC[l] + radiusRf_DIREC[l]) {
			prob_d = exp(-alpha1_direc * pow((r - radiusRs_DIREC[l]), beta1_direc) / pow((radiusRs_DIREC[l] + radiusRf_DIREC[l] - r),
				beta2_direc) + alpha2_direc);
		}
		else
			return 0.0;
		if (pan <= pan_range[l])
			prob_pan = 1.0;
		else if (pan > pan_range[l] && pan <= pan_ratio_mid * pan_range[l]) {
			prob_pan = 1 - vu_pan * exp(1 - pow((((pan_ratio_mid - 1.0) * pan_range[l]) / (pan - pan_range[l])), tau1_pan));
			//printf("%lf->%lf->%lf\n",((pan_ratio_mid-1.0)*pan_range[l]),pan-pan_range[l],prob_pan);
		}
		else if (pan > pan_ratio_mid * pan_range[l] && pan < pan_ratio_upp * pan_range[l]) {
			prob_pan = 0 + mu_pan * exp(1 - pow((((pan_ratio_upp - pan_ratio_mid) * pan_range[l]) / (pan_ratio_upp * pan_range[l] - pan)),
				tau2_pan));
			//printf("%lf->%lf->%lf\n",((pan_ratio_upp-pan_ratio_mid)*pan_range[l]),(pan_ratio_upp*pan_range[l]-pan),prob_pan);
		}
		else
			return 0.0;
		if (tilt <= tilt_range[l])
			prob_tilt = 1.0;
		else if (tilt > tilt_range[l] && tilt <= tilt_ratio_mid * tilt_range[l]) {
			prob_tilt = 1 - vu_tilt * exp(1 - pow((((tilt_ratio_mid - 1.0) * tilt_range[l]) / (tilt - tilt_range[l])), tau1_tilt));
			//printf("%lf->%lf->%lf\n",((tilt_ratio_mid-1.0)*tilt_range[l]),tilt-tilt_range[l],prob_tilt);
		}
		else if (tilt > tilt_ratio_mid * tilt_range[l] && tilt < tilt_ratio_upp * tilt_range[l]) {
			prob_tilt = 0 + mu_tilt * exp(1 - pow((((tilt_ratio_upp - tilt_ratio_mid) * tilt_range[l]) /
				(tilt_ratio_upp * tilt_range[l] - tilt)), tau2_tilt));
			//printf("%lf->%lf->%lf\n",((tilt_ratio_upp-tilt_ratio_mid)*tilt_range[l]),(tilt_ratio_upp*tilt_range[l]-tilt),prob_tilt);
		}
		else
			return 0.0;
		//printf("%lf->%lf\n",r-radiusRs_DIREC[l],prob_d);
		//printf("%lf->%lf->%lf\n",((pan_ratio_mid-1.0)*pan_range[l]),pan-pan_range[l],prob_pan);
		//printf("%lf->%lf->%lf\n",((tilt_ratio_mid-1.0)*tilt_range[l]),tilt-tilt_range[l],prob_tilt);
		return (prob_d * prob_pan * prob_tilt);
	}
	else {
		return 0.0;
	}
}

static void Qoc()
{
	int i, j, k, l;
	double m;
	for (i = 0; i < lon; i++) {
		for (j = 0; j < wid; j++) {
			for (k = 0; k < hig; k++) {
				m = 1.0;
				for (l = 0; l < N_DIREC; l++) {  //
					if (!posFlag_DIREC[l]) continue;
					if (posCEIL_SENSOR[l] && k > pos3D_DIREC[l][Z]) continue;
					m *= (1 + ga * Oq_DIREC(i, j, k, l));
				}
				m = (m - 1) / ga;
				if (m > oq_beta) {
					m = 1;
				}
				else {
					m = 0.0;
				}
				qoc[i][j][k] = m; //printf("%f\n",qoc[i][j][k]);
			}
		}
	}
	//for(i=0;i<lon/2;i++)
	//{
	// for(j=0;j<wid/2;j++)
	// {
	//    printf("%1.1f ",qoc[i][j]);
	// }
	//printf("\n");
	//}
}

static double Cover()//
{
	int i, j, k;
	double m = 0.0;
	Qoc();
	for (i = 0; i < lon; i++) {
		for (j = 0; j < wid; j++) {
			for (k = 0; k < hig; k++) {
				m += qoc[i][j][k];
			}
		}
	}
	return (m / (N - occupVol));
}

static double Lifetime()
{
	// initialization
	for (int i = 0; i < N_SENSOR; i++) {
		n_sn_rn_com[i] = 0;
	}
	for (int i = 0; i < N_RELAY; i++) {
		n_rn_rn_com[i] = 0;
		n_data_local_RELAY[i] = 0;
		n_data_hop_RELAY[i] = 0;
		energy_consumed_RELAY[i] = 0;
		LT_RELAY[i] = 0;
	}
	avg_dist_SENSOR_IWSN = 0.0;

	// sensor nodes
	for (int i = 0; i < N_SENSOR; i++) {
		double minD = 1.0e99;
		double tmpD;
		int ID = -1;
		for (int j = 0; j < N_RELAY; j++) {
			tmpD = range(pos3D_SENSOR[i][X], pos3D_SENSOR[i][Y], pos3D_SENSOR[i][Z],
				pos3D_RELAY[j][X], pos3D_RELAY[j][Y], pos3D_RELAY[j][Z]);
			if (tmpD * resIWSN < d_th_sn)
				n_sn_rn_com[i]++;
			if (tmpD < minD) {
				minD = tmpD;
				ID = j;
			}
		}
		if (ID == -1) {
			hopID_SENSOR[i] = N_RELAY;
			com_dist_SENSOR[i] = range(pos3D_SENSOR[i][X], pos3D_SENSOR[i][Y], pos3D_SENSOR[i][Z],
				SINK_X, SINK_Y, SINK_Z) * resIWSN;
		}
		else {
			hopID_SENSOR[i] = ID;
			com_dist_SENSOR[i] = minD * resIWSN;
			n_data_local_RELAY[ID] += l0;
		}
		avg_dist_SENSOR_IWSN += com_dist_SENSOR[i];
	}
	avg_dist_SENSOR_IWSN /= N_SENSOR;
	/*std_dist_SENSOR = 0.0;
	for (int i = 0; i < N_SENSOR; i++) {
	std_dist_SENSOR += (com_dist_SENSOR[i] - avg_dist_SENSOR) * (com_dist_SENSOR[i] - avg_dist_SENSOR);
	}
	std_dist_SENSOR = sqrt(std_dist_SENSOR / N_SENSOR);*/

	// relay nodes
	double dist2sink[N_RELAY];
	int    ID_RELAY[N_RELAY];
	for (int i = 0; i < N_RELAY; i++) {
		ID_RELAY[i] = i;
		dist2sink[i] = range(pos3D_RELAY[i][X], pos3D_RELAY[i][Y], pos3D_RELAY[i][Z],
			SINK_X, SINK_Y, SINK_Z);
	}
	for (int i = N_RELAY - 1; i > 0; i--) {
		for (int j = 0; j < i; j++) {
			if (dist2sink[j] < dist2sink[j + 1]) {
				double tmpD = dist2sink[j + 1];
				dist2sink[j + 1] = dist2sink[j];
				dist2sink[j] = tmpD;
				int tmpID = ID_RELAY[j + 1];
				ID_RELAY[j + 1] = ID_RELAY[j];
				ID_RELAY[j] = tmpID;
			}
		}
	}

	for (int i = 0; i < N_RELAY; i++) {
		int ID = ID_RELAY[i];
		n_data_fwd_RELAY[ID] = (1.0 - d_DA) * n_data_local_RELAY[ID] + n_data_hop_RELAY[ID];
		double minD = dist2sink[i];
		int tmpID = -1;
		for (int j = i + 1; j < N_RELAY; j++) {
			int id = ID_RELAY[j];
			double tmpD = range(pos3D_RELAY[ID][X], pos3D_RELAY[ID][Y], pos3D_RELAY[ID][Z],
				pos3D_RELAY[id][X], pos3D_RELAY[id][Y], pos3D_RELAY[id][Z]);
			if (tmpD * resIWSN < d_th_rn) {
				n_rn_rn_com[ID]++;
				n_rn_rn_com[id]++;
			}
			if (tmpD < minD) {
				minD = tmpD;
				tmpID = id;
			}
		}
		if (tmpID == -1) {
			hopID_RELAY[ID] = N_RELAY;
			com_dist_RELAY[ID] = dist2sink[i] * resIWSN;
		}
		else {
			hopID_RELAY[ID] = tmpID;
			n_data_hop_RELAY[tmpID] += n_data_fwd_RELAY[ID];
			com_dist_RELAY[ID] = minD * resIWSN;
		}
	}

	double energy_max = -1.0;
	double LT_max = -1.0;
	for (int i = 0; i < N_RELAY; i++) {
		energy_consumed_RELAY[i] = (n_data_local_RELAY[i]) * E_DA +
			(n_data_local_RELAY[i] + n_data_hop_RELAY[i]) * E_elec;
		if (com_dist_RELAY[i] < d_th) {
			energy_consumed_RELAY[i] += n_data_fwd_RELAY[i] * E_elec + n_data_fwd_RELAY[i] * e_fs * com_dist_RELAY[i] * com_dist_RELAY[i];
		}
		else {
			energy_consumed_RELAY[i] += n_data_fwd_RELAY[i] * E_elec + n_data_fwd_RELAY[i] * e_mp * com_dist_RELAY[i] * com_dist_RELAY[i] *
				com_dist_RELAY[i] * com_dist_RELAY[i];
		}
		if (energy_consumed_RELAY[i] > energy_max) {
			energy_max = energy_consumed_RELAY[i];
		}

		if (energy_consumed_RELAY[i] > 0.0) {
			LT_RELAY[i] = energy_ini / energy_consumed_RELAY[i];
			if (LT_max < LT_RELAY[i])
				LT_max = LT_RELAY[i];
		}
	}

	LT_avg = 0.0;
	LT_std_IWSN = 0.0;
	for (int i = 0; i < N_RELAY; i++) {
		if (energy_consumed_RELAY[i] <= 0.0)
			LT_RELAY[i] = LT_max;
		LT_avg += LT_RELAY[i];
	}

	LT_avg /= N_RELAY;
	for (int i = 0; i < N_RELAY; i++) {
		LT_std_IWSN += (LT_RELAY[i] - LT_avg) * (LT_RELAY[i] - LT_avg);
	}
	LT_std_IWSN = sqrt(LT_std_IWSN / N_RELAY);

	if (energy_max <= 0.0)
		LT_min_IWSN = 1e-6;
	else
		LT_min_IWSN = energy_ini / energy_max;

	return (1.0e4 / LT_min_IWSN);
}

static double Reliability()
{
	double vPenalty = 0.0;
	n_relia_p = 0;
	double tmpD;
	double tmpS;

	for (int i = 0; i < N_SENSOR; i++) {
		for (int j = 0; j < N_RELAY; j++) {
			distNode2Relay[i][j] = range(pos3D_SENSOR[i][X], pos3D_SENSOR[i][Y], pos3D_SENSOR[i][Z],
				pos3D_RELAY[j][X], pos3D_RELAY[j][Y], pos3D_RELAY[j][Z]) * resIWSN;
		}
		tmpS = 0.0;
		for (int j = 0; j < n_rn_min; j++) {
			for (int k = j + 1; k < N_RELAY; k++) {
				if (distNode2Relay[i][j] > distNode2Relay[i][k]) {
					tmpD = distNode2Relay[i][j];
					distNode2Relay[i][j] = distNode2Relay[i][k];
					distNode2Relay[i][k] = tmpD;
				}
			}
			tmpS += distNode2Relay[i][j];
		}
		vPenalty += tmpS / n_rn_min;
	}
	for (int i = 0; i < N_RELAY; i++) {
		for (int j = 0; j < N_RELAY; j++) {
			distNode2Relay[N_SENSOR + i][j] = range(pos3D_RELAY[i][X], pos3D_RELAY[i][Y], pos3D_RELAY[i][Z],
				pos3D_RELAY[j][X], pos3D_RELAY[j][Y], pos3D_RELAY[j][Z]) * resIWSN;
		}
		distNode2Relay[N_SENSOR + i][i] = 1e99;
		tmpS = 0.0;
		for (int j = 0; j < n_rn_min; j++) {
			for (int k = j + 1; k < N_RELAY; k++) {
				if (distNode2Relay[N_SENSOR + i][j] > distNode2Relay[N_SENSOR + i][k]) {
					tmpD = distNode2Relay[N_SENSOR + i][j];
					distNode2Relay[N_SENSOR + i][j] = distNode2Relay[N_SENSOR + i][k];
					distNode2Relay[N_SENSOR + i][k] = tmpD;
				}
			}
			tmpS += distNode2Relay[N_SENSOR + i][j];
		}
		vPenalty += tmpS / n_rn_min;
	}

	return (vPenalty / (15.0 * (N_SENSOR + N_RELAY)));
}

void Fitness_IWSN(double* individual, double* fitness, double* constrainV, int nx, int M)
{
	//if (!checkLimits_IWSN(individual, nx)) {
	//    printf("checkLimits_IWSN FAIL, exiting...\n");
	//    exit(-1);
	//}
	posCountBad = 0;
	for (int i = 0; i < N_DIREC; i++) {
		pan_angle[i] = individual[i * D_DIREC + PAN] * angE;
		tilt_angle[i] = angEmin +
			individual[i * D_DIREC + TILT] * (angEmax - angEmin);
		posFlag_DIREC[i] =
			TransPos_2D3D(1, tilt_angle[i],
				individual[i * D_DIREC + D1],
				individual[i * D_DIREC + D2],
				individual[i * D_DIREC + D3],
				&pos3D_DIREC[i][0],
				&posCEIL_SENSOR[i]);
		if (!posFlag_DIREC[i])
			posCountBad++;
		pos3D_SENSOR[i][X] = pos3D_DIREC[i][X];
		pos3D_SENSOR[i][Y] = pos3D_DIREC[i][Y];
		pos3D_SENSOR[i][Z] = pos3D_DIREC[i][Z];
		posFlag_SENSOR[i] = posFlag_DIREC[i];
	}
	int offset_D = N_DIREC * D_DIREC;
	for (int i = 0; i < N_RELAY; i++) {
		posFlag_RELAY[i] =
			TransPos_2D3D(0, 0.0,
				individual[offset_D + i * D_RELAY + D1],
				individual[offset_D + i * D_RELAY + D2],
				individual[offset_D + i * D_RELAY + D3],
				&pos3D_RELAY[i][0],
				NULL);
		if (!posFlag_RELAY[i])
			posCountBad++;
	}
	//printf("posCountBad:%d\n",posCountBad);
	fitness[0] = 1.0 - Cover() + (posCountBad)*penaltyVal;
	fitness[1] = Lifetime() + (posCountBad)*penaltyVal;
	fitness[2] = Reliability() + (posCountBad)*penaltyVal;

	//printf("n_relia_p:%d\n", n_relia_p);
	for (int i = 0; i < IWSNOBJ; i++) {
		fitness[i] += n_relia_p * penaltyVal;
	}
}

// static double Control_Angle(double a)//
// {
//     while (a >= 2 * pi || a < 0) {
//         if (a >= 2 * pi)
//             a = a - 2 * pi;
//         else
//             a = a + 2 * pi;
//     }
//     return a;
// }

static double ArcPan(int i, int j, int k, int a, int b, int c)
{
	double temp;
	if (j == b) {  //
		if (a > i)
			return 0.0;
		else if (a < i)
			return pi;
	}
	temp = (a - i) / sqrt((double)(a - i) * (a - i) + (b - j) * (b - j)); //
	if (j > b)
		return   2 * pi - acos(temp);
	else
		return acos(temp);
}

static double ArcTilt(int i, int j, int k, int a, int b, int c)
{
	double temp;
	if (k == c)  //
		return 0.0;
	temp = sqrt((double)(a - i) * (a - i) + (b - j) * (b - j)) /
		sqrt((double)(a - i) * (a - i) + (b - j) * (b - j) + (c - k) * (c - k)); //
	if (k > c)
		return -acos(temp);
	else
		return acos(temp);
}

// static double Myerf(double a)//a>0 erf(a)>0 (erf/2)
// {
//     int i, j;
//     double step, step_number, sum, t, temp;
//     if (a < 0) {
//         j = -1;
//         a = -a;
//     } else
//         j = 1;
//     step_number = 10; //
//     step = a / step_number; //
//     sum = 0.0;
//     for (i = 0; i < step_number; i++) {
//         t = (i + 0.5) * step;
//         temp = exp(-pow(t, 2)) * step;
//         if (temp < 0.000001)
//             break;
//         sum += temp;
//         //printf("%f\n",exp(-(pow(t,2)/2.0))*step);
//     }
//     //double result=0.5+j*(1.0/sqrt(2*pi))*sum;
//     //printf("%lf->%lf\n",j*a,result);
//     return j * (2.0 / sqrt(pi)) * sum;
// }