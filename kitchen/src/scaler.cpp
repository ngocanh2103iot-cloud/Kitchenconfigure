#include "scaler.h"

float scaler_mean[10] = {
-324.508301,
45.3350307,
19.9810577,
22.2974987,
5.52163486,
7.45852474,
1.83366689,
3.84607186,
0.806337962,
7.21899233
};

float scaler_std[10] = {
1083.9753,
51.8702187,
2.85822120,
15.3813329,
11.0022060,
10.6231585,
8.11144689,
9.47332870,
8.80352944,
4.65686272
};

void scale_features(float *features)
{
    for(int i=0;i<10;i++)
        features[i] =
        (features[i]-scaler_mean[i])/
        scaler_std[i];
}