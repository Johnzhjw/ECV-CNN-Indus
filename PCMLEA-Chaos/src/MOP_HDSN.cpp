#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "MOP_HDSN.h"

#define lon 32
#define wid 32
#define resolution 5
#define X 0
#define Y 1
#define PAN 2
#define TILT 3
#define PI 3.14
#define pi 3.1415926
#define r_min (45.0*2.0/3.0/resolution) // 45m//6
#define r_max (72.0*2.0/3.0/resolution) // 72m//9.6
#define r_ratio_mid 1.5
#define r_ratio_upp 2.0
#define vi1 0.5
#define vi2 0.5
#define omega1 3.6
#define omega2 3.6
#define P (lon*wid)//????μ?μ???êy
#define ga -0.5
#define oq_beta 0.9
#define emergy 200.0//10000.0;//3?ê??üá?
#define angle_min (pi/4*2/3/2)
#define angle_max (pi/3*2/3/2)
#define angle_ratio_mid 1.5
#define angle_ratio_upp 2.0
#define lamda1 0.5
#define lamda2 0.5
#define mu1 3.6
#define mu2 3.6
#define delta1_min 0.5
#define delta1_max 1.5
#define delta2_min 0.5
#define delta2_max 1.5
#define rrange 1
#define penalty 1e6
#define sqrtPI (sqrt(pi))
#define WHratio (1.33333)

static double map[lon][wid] = {
	{ 142.8873, 144.975, 147.2036, 149.4911, 151.5645, 153.6258, 155.4167, 156.9564, 158.523, 160.236, 161.5427, 162.4541, 163.1266, 163.7684, 164.6583, 165.7076, 166.9778, 168.3099, 169.6201, 170.5744, 171.2026, 171.634, 171.7931, 171.6509, 171.1994, 170.2342, 168.7902, 166.8209, 164.6255, 162.1241, 158.6438, 155.2589 },
	{ 140.9955, 143.2895, 145.2849, 147.4148, 149.5843, 151.809, 154.0269, 155.8413, 157.53, 159.3112, 160.8375, 162.2539, 163.2565, 164.2889, 165.3935, 166.6544, 168.172, 169.7021, 171.0673, 172.1726, 173.036, 173.6821, 174.0767, 174.121, 173.7551, 172.7876, 171.2936, 169.3275, 166.9454, 164.2433, 160.9538, 157.4299 },
	{ 138.3064, 141.0544, 143.112, 145.0917, 147.2261, 149.4449, 151.8214, 154.1107, 156.0865, 158.0539, 159.7833, 161.4601, 162.9837, 164.5596, 165.8477, 167.4011, 169.0785, 170.769, 172.3284, 173.6877, 174.8096, 175.7224, 176.3673, 176.5683, 176.1958, 175.1571, 173.6043, 171.6449, 169.2818, 166.5175, 163.1042, 159.3611 },
	{ 135.4902, 138.291, 140.6118, 142.6745, 144.8158, 147.0999, 149.3875, 151.7399, 154.022, 156.141, 158.2274, 160.2221, 162.3194, 164.4777, 166.1134, 167.8515, 169.6341, 171.4534, 173.3118, 174.9653, 176.3343, 177.4351, 178.1537, 178.3005, 178.0551, 177.0905, 175.6849, 173.7118, 171.4488, 168.667, 165.0878, 161.5374 },
	{ 133.1535, 135.4667, 137.8264, 140.1407, 142.4379, 144.8181, 147.1357, 149.3471, 151.6223, 153.9091, 156.269, 158.5893, 160.9562, 163.7943, 165.9491, 167.956, 169.8347, 171.7268, 173.8866, 175.8331, 177.4511, 178.4491, 179.0574, 179.2287, 179.0559, 178.5275, 177.3573, 175.418, 173.0889, 170.2852, 166.859, 163.3624 },
	{ 131.5225, 133.3638, 135.276, 137.4557, 140.0497, 142.6234, 145.041, 147.2531, 149.432, 151.6854, 154.1407, 156.6805, 159.2073, 161.9246, 164.846, 167.4958, 169.6873, 171.688, 174.0401, 176.2298, 178.1104, 179.0598, 179.782, 180.0914, 180.0969, 179.807, 178.8336, 176.6542, 174.2203, 171.2579, 168.1404, 164.9522 },
	{ 130.4845, 132.1995, 133.7579, 135.5313, 137.821, 140.6012, 143.2289, 145.5306, 147.697, 149.778, 151.9798, 154.5258, 157.1356, 159.8188, 162.8618, 166.192, 169.0683, 171.3707, 173.784, 176.1637, 178.2545, 179.4614, 180.2987, 180.6259, 180.633, 180.3147, 179.3144, 177.3426, 174.7659, 171.8251, 168.9632, 166.3671 },
	{ 130.2295, 131.6755, 133.1777, 134.7616, 136.6239, 139.0541, 141.7543, 144.2876, 146.3948, 148.4132, 150.1075, 152.1667, 154.8209, 157.6102, 160.7723, 164.2704, 167.855, 170.6921, 173.3238, 175.8044, 178.1611, 179.7671, 180.6967, 181.1521, 181.1592, 180.687, 179.8502, 177.7209, 175.1449, 172.5212, 170.094, 168.0994 },
	{ 130.6451, 131.7686, 133.1382, 134.6516, 136.3273, 138.329, 140.8246, 143.3662, 145.5602, 147.5734, 149.1204, 150.4466, 152.372, 155.5676, 159.0709, 162.7183, 166.4507, 170.0384, 172.8193, 175.4252, 177.922, 179.9083, 181.0366, 181.6319, 181.6165, 181.0638, 180.132, 178.0173, 175.7003, 173.6829, 171.6833, 170.3516 },
	{ 131.5329, 132.3542, 133.6369, 135.1144, 136.7319, 138.5613, 140.7311, 143.0792, 145.2505, 147.1842, 148.6293, 149.6755, 151.294, 154.2996, 158.1553, 161.9301, 165.7721, 169.4951, 172.497, 175.2752, 177.8429, 180.1128, 181.5151, 182.2467, 182.1888, 181.4501, 180.4579, 178.5222, 176.5343, 174.8235, 173.8397, 172.9631 },
	{ 132.8775, 133.2487, 134.5772, 135.9471, 137.521, 139.3696, 141.2529, 143.3812, 145.4198, 147.3151, 148.5496, 149.6691, 151.4606, 154.7269, 158.5063, 162.1952, 165.9558, 169.5135, 172.5751, 175.4679, 178.1004, 180.5262, 182.2745, 183.0451, 182.981, 182.2491, 181.1519, 179.5964, 177.7421, 176.8298, 176.1131, 175.9183 },
	{ 134.5579, 135.0639, 135.8825, 137.11, 138.6242, 140.387, 142.0826, 144.0225, 146.002, 147.9087, 149.1357, 150.3598, 152.7863, 156.2148, 159.7748, 163.2613, 166.7908, 170.1061, 173.0963, 175.9561, 178.6534, 181.243, 183.1164, 183.9537, 183.8402, 183.192, 182.3498, 181.1999, 180.1049, 179.5314, 179.5023, 180.222 },
	{ 136.7023, 137.1974, 137.7785, 138.6098, 139.9717, 141.469, 143.1403, 145.0087, 146.9913, 148.7007, 149.9999, 152.25, 155.2604, 158.435, 161.5972, 164.8068, 168.1015, 171.1158, 173.9849, 176.7477, 179.4884, 182.222, 184.1981, 184.9359, 185.0179, 184.3417, 183.8333, 183.2606, 182.9444, 182.9446, 183.4175, 184.4315 },
	{ 139.6931, 139.9455, 140.1879, 140.5283, 141.3981, 142.7369, 144.4099, 146.3031, 148.3638, 150.1245, 152.6119, 155.2796, 158.1284, 160.9957, 163.9427, 166.9005, 169.8057, 172.4585, 175.1694, 177.8511, 180.5806, 183.3041, 185.02, 185.8558, 185.8965, 185.3997, 185.1073, 185.1019, 185.4192, 185.9823, 186.6342, 187.7312 },
	{ 143.3035, 143.1833, 143.0737, 142.9647, 143.1912, 144.1708, 145.744, 147.8082, 149.9955, 152.6249, 155.3123, 158.1204, 160.7335, 163.6191, 166.6825, 169.5093, 171.733, 174.2842, 176.815, 179.3409, 181.9559, 184.4853, 186.1759, 187.0048, 187.0202, 186.65, 186.5155, 187.0286, 187.9307, 188.9599, 190.0735, 191.1185 },
	{ 147.3328, 146.894, 146.3295, 145.8094, 145.4467, 145.6214, 146.8871, 149.0482, 151.5715, 154.4266, 157.3432, 160.0929, 162.8143, 166.0915, 169.1783, 171.521, 173.8528, 176.3715, 178.8573, 181.2553, 183.7469, 185.9284, 187.498, 188.204, 188.3711, 188.1829, 188.427, 189.3531, 190.5424, 191.8044, 193.103, 194.3357 },
	{ 151.3907, 150.7403, 149.9614, 149.0052, 148.1223, 147.6212, 147.879, 149.9312, 152.4476, 155.3446, 158.3205, 161.1906, 164.1216, 167.3833, 170.4203, 173.0325, 175.7262, 178.4076, 180.8764, 183.3875, 185.6431, 187.6328, 188.9386, 189.6281, 189.8773, 189.8996, 190.3659, 191.4423, 192.6763, 194.2758, 195.8964, 197.341 },
	{ 155.2875, 154.3933, 153.2705, 152.1245, 151.1299, 150.391, 150.1458, 151.0235, 153.0868, 155.7588, 158.5711, 161.4731, 164.4799, 167.6618, 170.7356, 173.6265, 176.7587, 179.7986, 182.6486, 185.1477, 187.2737, 189.0931, 190.6413, 191.1708, 191.4539, 191.5907, 192.0991, 192.9165, 194.3057, 196.3029, 198.2765, 199.6792 },
	{ 159.0586, 157.9175, 156.5458, 155.2592, 154.1664, 153.3836, 152.9822, 153.1857, 154.2019, 156.1877, 158.6046, 161.2125, 164.0011, 167.2095, 170.318, 173.3624, 176.7282, 180.2783, 183.5963, 186.1461, 188.2787, 190.2852, 191.8625, 192.8434, 193.1086, 193.1103, 193.349, 194.1232, 195.7168, 197.916, 199.9641, 201.6257 },
	{ 162.5448, 161.2305, 159.7622, 158.4555, 157.3607, 156.6105, 156.1628, 156.0065, 156.332, 157.2724, 158.736, 160.6847, 163.0299, 166.146, 169.2232, 172.2917, 175.6614, 179.56, 183.1984, 186.0143, 188.4149, 190.7268, 192.9987, 194.4361, 194.7275, 194.677, 194.7961, 195.4847, 197.1069, 199.1427, 201.3737, 203.3545 },
	{ 165.6769, 164.1655, 162.767, 161.4905, 160.5061, 159.9191, 159.5697, 159.351, 159.2811, 159.371, 159.6286, 160.4796, 162.1469, 164.731, 167.8892, 170.8493, 173.9254, 177.9491, 181.992, 184.9487, 187.7091, 190.2824, 192.9373, 194.9747, 196.0858, 196.3105, 196.4597, 197.0276, 198.324, 199.9283, 202.4532, 204.6808 },
	{ 168.96, 167.1395, 165.6623, 164.5997, 163.7843, 163.2083, 162.9767, 162.8868, 162.8061, 162.6879, 162.5205, 162.5125, 162.9494, 164.2838, 166.8752, 169.6362, 172.1519, 176.0833, 180.3182, 183.6687, 186.6737, 189.4842, 192.3124, 194.9899, 196.9124, 197.7587, 198.1152, 198.3922, 199.1624, 200.6911, 203.1698, 205.5797 },
	{ 172.2743, 170.3106, 168.7787, 167.8842, 167.2089, 166.6662, 166.4278, 166.3914, 166.3181, 166.0768, 165.771, 165.5659, 165.6199, 166.043, 167.2241, 169.2511, 171.3715, 174.7616, 178.8629, 182.4494, 185.7319, 188.826, 191.8611, 194.8768, 197.5484, 198.7404, 199.1643, 199.3466, 199.8182, 201.2308, 203.6424, 206.1464 },
	{ 176.0957, 173.94, 172.2791, 171.2609, 170.6474, 170.2372, 170.0259, 169.9308, 169.8048, 169.7379, 169.336, 169.0524, 168.8964, 168.9494, 169.4099, 170.4891, 171.9529, 174.7549, 178.1285, 181.601, 185.1199, 188.533, 191.7779, 194.9655, 198.0199, 199.5461, 200.1518, 200.2701, 200.6277, 201.5743, 203.9402, 206.4411 },
	{ 179.9467, 178.0288, 176.484, 175.4261, 174.6554, 174.0247, 173.6203, 173.337, 173.1247, 172.9423, 172.7128, 172.4781, 172.2808, 172.196, 172.4271, 173.2037, 174.4658, 176.283, 178.7548, 181.6946, 185.268, 188.8326, 192.2208, 195.4634, 198.385, 200.663, 200.9074, 201.0065, 201.324, 201.8509, 204.0864, 206.5229 },
	{ 184.0142, 182.3206, 180.8738, 179.8535, 179.1125, 178.535, 177.9819, 177.3572, 176.8102, 176.4561, 176.2211, 176.0158, 175.8817, 175.8545, 176.0249, 176.486, 177.2777, 178.6446, 180.7057, 183.5169, 186.7601, 190.0538, 193.2982, 196.4551, 199.3083, 201.0476, 201.6778, 201.7269, 202, 202.1924, 204.1565, 206.4762 },
	{ 187.9319, 186.9768, 185.7487, 184.7165, 183.9492, 183.3218, 182.71, 182.0636, 181.3649, 180.8411, 180.5477, 180.268, 179.8834, 179.6577, 179.6673, 179.8888, 180.4929, 181.6829, 183.9588, 186.6218, 189.4165, 192.2384, 195.0714, 197.974, 200.6043, 202.2575, 202.6908, 202.6034, 202.4319, 202.7958, 204.1551, 206.3507 },
	{ 190.6106, 190.7771, 190.4228, 189.6696, 188.8991, 188.1682, 187.4838, 186.873, 186.3445, 185.8326, 185.2939, 184.7778, 184.3103, 184.0465, 184.0625, 184.3365, 185.0035, 186.3161, 188.272, 190.4258, 192.6859, 195.0176, 197.573, 200.1739, 202.3237, 203.4604, 203.7563, 203.5312, 203.4407, 203.4634, 204.1699, 206.1973 },
	{ 192.5234, 193.387, 194.001, 194.1382, 193.9169, 193.3623, 192.6953, 192.065, 191.5429, 191.0222, 190.4941, 189.9677, 189.4395, 189.0904, 188.9614, 189.0047, 189.6441, 190.7843, 192.5791, 194.6364, 196.3422, 198.3354, 200.5071, 202.661, 204.1957, 204.7467, 204.3709, 204, 204, 204, 204.0485, 205.9162 },
	{ 193.9254, 195.0381, 195.9854, 196.6965, 197.1917, 197.6031, 197.5441, 197.1664, 196.7175, 196.3191, 195.962, 195.4266, 194.7402, 194.0555, 193.4354, 193.253, 193.6545, 195.0529, 196.7608, 198.3274, 200.1245, 201.9572, 203.6367, 204.5722, 205.3501, 205.8555, 204.8672, 203.8587, 203.7443, 203.6299, 203.7901, 205.483 },
	{ 194.5631, 196.0016, 197.1938, 198.3002, 199.2309, 199.8874, 200.1592, 200.2171, 200.2102, 200.1527, 200.0758, 199.8437, 199.3666, 198.8146, 198.2436, 197.9442, 198.2328, 199.2774, 200.6716, 202.2484, 203.8833, 204.4681, 205.1123, 205.783, 206, 205.9237, 204.8109, 203.6721, 202.7153, 202.5421, 203.1864, 204.8036 },
	{ 194.7748, 196.5845, 197.9857, 199.3015, 200.4037, 201.1567, 201.8294, 202.208, 202.4561, 202.7549, 202.9314, 202.9856, 202.9216, 202.7655, 202.6152, 202.4856, 202.6098, 203.2817, 204.1547, 204.4098, 204.8519, 205.6015, 206.0481, 206, 206, 205.6275, 204.5165, 203.158, 201.98, 201.3263, 201.8687, 203.7766 }
};

static double qoc[wid][lon];
static double radiusR_low[ds];
static double radiusR_mid[ds];
static double radiusR_upp[ds];
static double angle_range_low[ds];
static double angle_range_mid[ds];
static double angle_range_upp[ds];
static double com_r[ds];
static double delta1[ds];
static double delta2[ds];

static double myDistance[ds][ds];
static int connectstatus[ds][ds];
static double maxAlt = -1e6, minAlt = 1e6;
static double maxFluct = -1e6;
static int maxConnection;//最大传感器连通数

/***************************************************/
//oˉêyéù?÷
// static int RandomInteger(int low, int high);
static double range(int i, int j, int a, int b); //?????àà?
static int LOS(int i, int j, int a, int b);
static double Oq(int i, int j, int a, int b, double pan, double tilt, int k); //?????ì2a???ê_new
static void Qoc(double* a);//??D?????QOC???ó
static double Cover(double* a);//?????¨μ??2???ê
// static double Energy(double* a);//????ê±??
static double Uniformity(double* a);//á?í¨D?//conflict with lifetime --- omitted
static double DeploymentCost(double* a);//2?êe·?ó?
// static double Control_Angle(double a);//???è′|àíoˉêy
static double ArcPan(int i, int j, int a, int b); //·μ??????μ?ó?′??D?÷μ?μ??±??1?óúX?áμ????è_new
static double ArcTilt(int i, int j, int a, int b); //tilt angle
// static double myErf(double a);
/***************************************************/
/*
int main()
{
int i,j,k,m,n;
srand(36);
for(k=0;k<ds;k++)
{
radiusR_low[k]=r_min+rand()/(RAND_MAX+1.0)*(r_max-r_min);printf("%lf ",radiusR_low[k]);
radiusR_mid[k]=r_ratio_mid*radiusR_low[k];
radiusR_upp[k]=r_ratio_upp*radiusR_low[k];
angle_range_low[k]=angle_min+rand()/(RAND_MAX+1.0)*(angle_max-angle_min);printf("%lf ",angle_range_low[k]);
angle_range_mid[k]=angle_ratio_mid*angle_range_low[k];
angle_range_upp[k]=angle_ratio_upp*angle_range_low[k];
com_r[k]=1.1*(radiusR_upp[k]);printf("%lf ",com_r[k]);
delta1[k]=1.0;printf("%lf ",delta1[k]);
delta2[k]=WHratio*delta1[k];printf("%lf\n",delta2[k]);
}
double fluct;
int np;
for(i=0;i<lon;i++)
{
for(j=0;j<wid;j++)
{
if(maxAlt<map[i][j])
maxAlt=map[i][j];
if(minAlt>map[i][j])
minAlt=map[i][j];
fluct=0.0;
np=0;
for(m=-rrange;m<=rrange;m++)
{
for(n=-rrange;n<=rrange;n++)
{
if(m==0&&n==0) continue;
if(i+m>=0&&i+m<lon&&j+n>=0&&j+n<wid)
{
fluct+=fabs(map[i][j]-map[i+m][j+n]);
np++;
}
}
}
fluct/=np;
if(maxFluct<fluct)
maxFluct=fluct;
}
}
srand((int)time(NULL));
double individual[ds*UNIT];
double fitness[HDSNOBJ];
for(k=0;k<ds;k++)
{
individual[k*UNIT+X]=RandomInteger(0,lon-1);
individual[k*UNIT+Y]=RandomInteger(0,wid-1);
individual[k*UNIT+PAN]=RandomInteger(0,719)/360.0*pi;
individual[k*UNIT+TILT]=RandomInteger(0,719)/360.0*pi;
}
Fitness_HDSN(individual,fitness);
printf("coverage:%lf\n",fitness[0]);
//printf("lifetime:%lf\n",fitness[1]);
printf("uniformity:%lf\n",fitness[1]);
printf("deploymentCost:%lf\n",fitness[2]);
return 0;
}
*/

void SetLimits_HDSN(double* minLimit, double* maxLimit, int nx)
{
	int i, j, k, m, n;
	for (k = 0; k < ds; k++) {
		minLimit[k * UNIT + X] = 0;
		minLimit[k * UNIT + Y] = 0;
		minLimit[k * UNIT + PAN] = 0;
		minLimit[k * UNIT + TILT] = -pi / 2.0;
		maxLimit[k * UNIT + X] = lon - 1e-10;
		maxLimit[k * UNIT + Y] = wid - 1e-10;
		maxLimit[k * UNIT + PAN] = 2 * pi - 1e-10;
		maxLimit[k * UNIT + TILT] = pi / 2.0;
	}
	double fluct;
	int np;
	for (i = 0; i < lon; i++) {
		for (j = 0; j < wid; j++) {
			if (maxAlt < map[i][j])
				maxAlt = map[i][j];
			if (minAlt > map[i][j])
				minAlt = map[i][j];
			fluct = 0.0;
			np = 0;
			for (m = -rrange; m <= rrange; m++) {
				for (n = -rrange; n <= rrange; n++) {
					if (m == 0 && n == 0) continue;
					if (i + m >= 0 && i + m < lon && j + n >= 0 && j + n < wid) {
						fluct += fabs(map[i][j] - map[i + m][j + n]);
						np++;
					}
				}
			}
			fluct /= np;
			if (maxFluct < fluct)
				maxFluct = fluct;
		}
	}
	srand(36);
	for (k = 0; k < ds; k++) {
		radiusR_low[k] = r_min + rand() / (RAND_MAX + 1.0) * (r_max - r_min);
		//printf("%lf ", radiusR_low[k]);
		radiusR_mid[k] = r_ratio_mid * radiusR_low[k];
		radiusR_upp[k] = r_ratio_upp * radiusR_low[k];
		angle_range_low[k] = angle_min + rand() / (RAND_MAX + 1.0) * (angle_max - angle_min);
		//printf("%lf ", angle_range_low[k]);
		angle_range_mid[k] = angle_ratio_mid * angle_range_low[k];
		angle_range_upp[k] = angle_ratio_upp * angle_range_low[k];
		com_r[k] = 1.1 * (radiusR_upp[k]);
		//printf("%lf ", com_r[k]);
		delta1[k] = 1.0;
		//printf("%lf ", delta1[k]);
		delta2[k] = WHratio * delta1[k];
		//printf("%lf\n", delta2[k]);
	}
}

int CheckLimits_HDSN(double* x, int nx)
{
	int k;

	for (k = 0; k < ds; k++) {
		if (x[k * UNIT + X] < 0 || x[k * UNIT + X] > lon - 1e-10) {
			printf("Check limits FAIL - HDSN: %d, %.16e not in [%.16e, %.16e]\n", k * UNIT + X, x[k * UNIT + X], 0.0, lon - 1e-10);
			return false;
		}
		if (x[k * UNIT + Y] < 0 || x[k * UNIT + Y] > wid - 1e-10) {
			printf("Check limits FAIL - HDSN: %d, %.16e not in [%.16e, %.16e]\n", k * UNIT + Y, x[k * UNIT + Y], 0.0, wid - 1e-10);
			return false;
		}
		if (x[k * UNIT + PAN] < 0 || x[k * UNIT + PAN] > 2 * pi - 1e-10) {
			printf("Check limits FAIL - HDSN: %d, %.16e not in [%.16e, %.16e]\n", k * UNIT + PAN, x[k * UNIT + PAN], 0.0, 2 * pi - 1e-10);
			return false;
		}
		if (x[k * UNIT + TILT] < -pi / 2.0 || x[k * UNIT + TILT] > pi / 2.0) {
			printf("Check limits FAIL - HDSN: %d, %.16e not in [%.16e, %.16e]\n", k * UNIT + TILT, x[k * UNIT + TILT], -pi / 2.0, pi / 2.0);
			return false;
		}
	}

	return true;
}

// static int RandomInteger(int low, int high)
// {
//     return (low + (int)(rand() / (RAND_MAX + 1.0) * (high - low + 1)));
// }

static double range(int i, int j, int a, int b) //?????àà?
{
	double r;
	r = sqrt((double)((i - a) * (i - a) + (j - b) * (j - b) + (map[i][j] - map[a][b]) / resolution *
		(map[i][j] - map[a][b]) / resolution));
	return r;
}
static int LOS(int i, int j, int a, int b)                            ///LOSoˉêy
{
	double h = map[i][j];
	double cc = map[a][b];
	int m;
	int x, y, x1, y1;
	double z1, z2,
		z;     //z空间直线上点高度   x、y为两点横纵坐标差（正整数＿x1为单位差量（带符号） z1、z2地形的坐标点高度
	double k, k1;         //两点成线的斜玿
	double x2;            //用于存储所求点坐标
	double y2;
	int x3;
	int y3;               //将y2整形匿
	x = abs(a - i);       //俩点间的坐标巿
	y = abs(b - j);
	if (x >= y) {
		if (i != a) {
			x1 = (a - i) / x;         //存正负数
			k = ((double)(b - j)) / ((double)(a - i));
			k1 = ((double)(cc - h)) / (a - i);
			for (m = 1; m < x; m++) {
				x2 = i + x1 * m;
				y2 = k * (x2 - i) + j;
				z = k1 * (x2 - i) + h; ////x、y、z坐标（直线）
				if (y2 == (int)y2) {
					x3 = (int)x2;
					y3 = (int)y2;
					z1 = map[x3][y3];
					if (z < z1) {
						return 0;
					}
				}
				else {
					x3 = (int)x2;
					y3 = (int)y2;
					z1 = map[x3][y3];
					y3 = y3 + 1;
					z2 = map[x3][y3];
					if (z < z1 || z < z2) {
						return 0;
					}
				}
			}
		}
	}
	else {
		if (j != b) {
			y1 = (b - j) / y;         //存正负数
			k = ((double)(a - i)) / ((double)(b - j));
			k1 = ((double)(cc - h)) / (b - j);
			for (m = 1; m < y; m++) {
				y2 = j + y1 * m;
				x2 = k * (y2 - j) + i;
				z = k1 * (y2 - j) + h; ////x、y、z坐标（直线）
				if (x2 == (int)x2) {
					x3 = (int)x2;
					y3 = (int)y2;
					z1 = map[x3][y3];
					if (z < z1) {
						return 0;
					}
				}
				else {
					x3 = (int)x2;
					y3 = (int)y2;
					z1 = map[x3][y3];
					x3 = x3 + 1;
					z2 = map[x3][y3];
					if (z < z1 || z < z2) {
						return 0;
					}
				}
			}
		}
	}
	return 1;
}

static double Oq(int i, int j, int a, int b, double pan, double tilt, int k) //′??D?÷????ó?????μ?,an?a[0,2pi)
{
	double r;        //ó?óú′?·???±ê????ó???±ê′??D?÷μ??àà?range (s,p)
	if (i == a && j == b)  //á?μ???o?￡??2???è?a1
		return 1.0;
	if (LOS(i, j, a, b)) {
		r = range(i, j, a, b);
		pan = pan - ArcPan(i, j, a, b); //an?a(-2pi,2pi)
		if (pan < 0)
			pan = -pan; //[0,2pi)
		if (pan > pi)
			pan = 2 * pi - pan; //an为[0,pi]，像素点相对传感器的偏转角
		//printf("%f\t",pan);
		tilt = tilt - ArcTilt(i, j, a, b);
		if (tilt < 0)
			tilt = -tilt;
		if (tilt > pi)
			tilt = 2 * pi - tilt;
		//printf("%f\n",tilt);
		double prob_d, prob_angle1, prob_angle2, angle_mix;
		if (r <= radiusR_low[k])
			prob_d = 1.0;
		else if (r > radiusR_low[k] && r <= radiusR_mid[k]) {
			prob_d = 1 - vi1 * exp(tan(pi / 2.0 * (pow((r - radiusR_low[k]) / (radiusR_mid[k] - radiusR_low[k]), omega1) - 1.0)));
			//printf("mid->%lf->%lf->%lf\n",r - radiusR_low[k],radiusR_mid[k] - radiusR_low[k],prob_d);
		}
		else if (r > radiusR_mid[k] && r < radiusR_upp[k]) {
			prob_d = 0 + vi2 * exp(tan(pi / 2.0 * (pow((radiusR_upp[k] - r) / (radiusR_upp[k] - radiusR_mid[k]), omega2) - 1.0)));
			//printf("upp->%lf->%lf->%lf\n",radiusR_upp[k] - r,radiusR_upp[k] - radiusR_mid[k],prob_d);
		}
		else
			prob_d = 0.0;

		angle_mix = pan / delta1[k];
		if (angle_mix <= angle_range_low[k])
			prob_angle1 = 1.0;
		else if (angle_mix > angle_range_low[k] && angle_mix <= angle_range_mid[k]) {
			prob_angle1 = 1 - lamda1 * exp(tan(pi / 2.0 * (pow((angle_mix - angle_range_low[k]) / (angle_range_mid[k] - angle_range_low[k]),
				mu1) - 1.0)));
			//printf("mid->%lf->%lf->%lf\n",angle_mix - angle_range_low[k],angle_range_mid[k] - angle_range_low[k],prob_angle1);
		}
		else if (angle_mix > angle_range_mid[k] && angle_mix < angle_range_upp[k]) {
			prob_angle1 = 0 + lamda2 * exp(tan(pi / 2.0 * (pow((angle_range_upp[k] - angle_mix) / (angle_range_upp[k] - angle_range_mid[k]),
				mu2) - 1.0)));
			//printf("upp->%lf->%lf->%lf\n",angle_range_upp[k] - angle_mix,angle_range_upp[k] - angle_range_mid[k],prob_angle1);
		}
		else
			prob_angle1 = 0.0;
		//printf("%lf\t",prob_d);
		//printf("%lf\t",prob_angle1);
		angle_mix = tilt / delta2[k];
		if (angle_mix <= angle_range_low[k])
			prob_angle2 = 1.0;
		else if (angle_mix > angle_range_low[k] && angle_mix <= angle_range_mid[k]) {
			prob_angle2 = 1 - lamda1 * exp(tan(pi / 2.0 * (pow((angle_mix - angle_range_low[k]) / (angle_range_mid[k] - angle_range_low[k]),
				mu1) - 1.0)));
			//printf("mid->%lf->%lf->%lf\n",angle_mix - angle_range_low[k],angle_range_mid[k] - angle_range_low[k],prob_angle2);
		}
		else if (angle_mix > angle_range_mid[k] && angle_mix < angle_range_upp[k]) {
			prob_angle2 = 0 + lamda2 * exp(tan(pi / 2.0 * (pow((angle_range_upp[k] - angle_mix) / (angle_range_upp[k] - angle_range_mid[k]),
				mu2) - 1.0)));
			//printf("upp->%lf->%lf->%lf\n",angle_range_upp[k] - angle_mix,angle_range_upp[k] - angle_range_mid[k],prob_angle2);
		}
		else
			prob_angle2 = 0.0;
		//printf("%lf\t",prob_d);
		//printf("%lf\t",prob_angle2);
		return (prob_d * prob_angle1 * prob_angle2);
	}
	else {
		return 0.0;
	}
}

static void Qoc(double* a)
{
	int i, j, k;
	double m;
	for (i = 0; i < lon; i++) {
		for (j = 0; j < wid; j++) {
			for (k = 0, m = 1.0; k < ds; k++) {  //±??ˉ2
				m *= (1 + ga * Oq((int)(a[k * UNIT + X]), (int)(a[k * UNIT + Y]), i, j, a[k * UNIT + PAN], a[k * UNIT + TILT], k));
			}
			m = (m - 1) / ga;
			if (m > oq_beta) {
				m = 1;
			}
			else {
				m = 0.0;
			}
			qoc[i][j] = m; //printf("%f\n",qoc[i][j]);
		}
	}
	//for(i=0;i<lon/2;i++)
	//{
	//  for(j=0;j<wid/2;j++)
	//  {
	//      printf("%1.1f ",qoc[i][j]);
	//  }
	//printf("\n");
	//}
}

static double Cover(double* a)//?????2???êcover
{
	int i, j;
	double m = 0.0;
	Qoc(a);
	for (i = 0; i < lon; i++)
		for (j = 0; j < wid; j++) {
			m += qoc[i][j];
		}
	return (m / P);
}

static double Uniformity(double* a)
{
	double connect = 0.0;
	int i, j;
	double di;
	int count = 0;
	for (i = 0; i < ds; i++) {
		for (j = i + 1; j < ds; j++) {
			myDistance[i][j] = myDistance[j][i] =
				di = range((int)(a[i * UNIT + X]), (int)(a[i * UNIT + Y]), (int)(a[j * UNIT + X]), (int)(a[j * UNIT + Y]));
			if (di < (com_r[i] < com_r[j] ? com_r[i] : com_r[j])) {
				connectstatus[i][j] = connectstatus[j][i] = 1;
				//count++;
			}
			else
				connectstatus[i][j] = connectstatus[j][i] = 0;
		}
		myDistance[i][i] = 0.0;
		connectstatus[i][i] = 1;
	}
	/*  printf("%lf\n",count/(1.0*ds*(ds-1.0)/2));
		for(i=0;i<ds;i++)
		{
		for(j=0;j<ds;j++)
		{
		printf("%lf\t",myDistance[i][j]);
		}
		printf("\n");
		}
		for(i=0;i<ds;i++)
		{
		for(j=0;j<ds;j++)
		{
		printf("%d ",connectstatus[i][j]);
		}
		printf("\n");
		}*/
	int flag[ds];
	int checked[ds];
	for (i = 0; i < ds; i++) flag[i] = 0;
	int status;
	int loop = 1;
	int cur_loop = -1;
	int pivot;
	for (i = 0; i < ds; i++) {
		if (flag[i] == 0) {
			count = 1;
			status = 1;
			flag[i] = loop;
			pivot = i;
			for (j = 0; j < ds; j++) checked[j] = 0;
			while (status) {
				checked[pivot] = 1; //visited
				for (j = 0; j < ds; j++) {
					if (flag[j] == 0 && connectstatus[pivot][j]) {
						flag[j] = loop;
						count++;
					}
				}
				status = 0;
				for (j = 0; j < ds; j++)
					if (flag[j] == loop && checked[j] == 0) { //unvisited
						pivot = j;
						status = 1;
						break;
					}
			}
			loop++;
			if (count > connect) {
				connect = count;
				cur_loop = loop - 1;
			}
		}
	}
	// int edges = 0;
	// double length = 0.0;
	int n_edge[ds] = { 0 }; //for(i=0;i<ds;i++)printf("%d\n",n_edge[i]);
	int sum_edge = 0;
	for (i = 0; i < ds; i++) {
		if (flag[i] == cur_loop) {
			for (j = 0; j < ds; j++) {
				if (connectstatus[i][j]) {
					n_edge[i]++;
					sum_edge++;
				}
			}
		}
	}
	double mean_edge = (double)sum_edge / connect;
	double std = 0.0;
	for (i = 0; i < ds; i++) {
		if (flag[i] == cur_loop) {
			std += fabs(n_edge[i] - mean_edge);
		}
	}
	std /= connect;
	std /= ds; //printf("\n%f\n",connect);
	maxConnection = (int)connect;
	return (std);
}
static double DeploymentCost(double* a)
{
	double depC = 0.0;
	int i, j, k, m, n;
	double fluct;
	int np;
	for (k = 0; k < ds; k++) {
		fluct = 0.0;
		np = 0;
		i = (int)a[k * UNIT + X];
		j = (int)a[k * UNIT + Y];
		for (m = -rrange; m <= rrange; m++) {
			for (n = -rrange; n <= rrange; n++) {
				if (m == 0 && n == 0) continue;
				if (i + m >= 0 && i + m < lon && j + n >= 0 && j + n < wid) {
					fluct += fabs(map[i][j] - map[i + m][j + n]);
					np++;
				}
			}
		}
		depC += fluct / (np * maxFluct);
		depC += (map[i][j] - minAlt) / (maxAlt - minAlt);
	}
	return (depC / ds / 2.0);
}

void Fitness_HDSN(double* individual, double* fitness, double* constrainV, int nx, int M)
{
	fitness[1] = Uniformity(individual);
	fitness[1] += (ds - maxConnection) * penalty;
	fitness[0] = 1.0 - Cover(individual) + (ds - maxConnection) * penalty;
	//fitness[1]=1.0-Energy(individual);
	fitness[2] = DeploymentCost(individual) + (ds - maxConnection) * penalty;
}

// static double Control_Angle(double a)//???è′|àíoˉêy
// {
//     while (a >= 2 * pi || a < 0) {
//         if (a >= 2 * pi)
//             a = a - 2 * pi;
//         else
//             a = a + 2 * pi;
//     }
//     return a;
// }

static double ArcPan(int i, int j, int a, int b)
{
	double temp;
	if (j == b) {
		if (a > i)
			return 0.0;
		else if (a < i)
			return pi;
	}
	temp = (a - i) / sqrt((double)(a - i) * (a - i) + (b - j) * (b - j));
	if (j > b)
		return 2 * pi - acos(temp);
	else
		return acos(temp);
}
static double ArcTilt(int i, int j, int a, int b)
{
	double temp;
	double hs, hp;
	hs = map[i][j];
	hp = map[a][b];
	if (hs == hp)  //·μ?????è0
		return 0.0;
	temp = sqrt((double)(a - i) * (a - i) + (b - j) * (b - j)) /
		sqrt((double)(a - i) * (a - i) + (b - j) * (b - j) + (hp - hs) * (hp - hs)); //?óóà?ò
	if (hs > hp)
		return -acos(temp);
	else
		return acos(temp);
}
// static double myErf(double a)//a>0ê±myErf(a)>0 (myErf/2)
// {
//     int i, j;
//     double step, step_number, sum, t, temp;
//     if (a < 0) {
//         j = -1;
//         a = -a;
//     } else
//         j = 1;
//     step_number = 10; //步数
//     step = a / step_number; //步长
//     sum = 0.0;
//     for (i = 0; i < step_number; i++) {
//         t = (i + 0.5) * step;
//         temp = exp(-t * t) * step;
//         if (temp < 0.000001)
//             break;
//         sum += temp;
//     }
//     return j * (2.0 / sqrtPI) * sum;
// }