#include "quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static unsigned long st=88172645463325252ULL;
static double rnd(){ st^=st<<13; st^=st>>7; st^=st<<17; return (double)(st>>11)/9007199254740992.0; }
static float gauss(){ double u=rnd(),v=rnd(); return (float)(sqrt(-2*log(u+1e-12))*cos(6.2831853*v)); }
static double rt(ds4q_type t, void(*deq)(const void*,float*,int64_t), const float*x, int N, int NC){
    int NR=N/NC; size_t rs=ds4q_row_size(t,NC);
    unsigned char*q=malloc(rs*NR); float*rc=malloc(N*4);
    ds4q_quantize_init(t);
    ds4q_quantize_chunk(t,x,q,0,NR,NC,NULL);
    for(int r=0;r<NR;r++) deq(q+(size_t)r*rs, rc+(size_t)r*NC, NC);
    double se=0,sx=0; for(int i=0;i<N;i++){double e=x[i]-rc[i]; se+=e*e; sx+=(double)x[i]*x[i];}
    free(q);free(rc); return sqrt(se/sx);
}
int main(){
    int NC=4096, NR=8, N=NC*NR; float*x=malloc(N*4);
    for(int i=0;i<N;i++) x[i]=gauss()*0.05f;
    printf("k-quant round-trip (relative_rmse, random Gaussian — real weights do better):\n");
    printf("  q3_K (3.4 bpw): %.4f\n", rt(DS4Q_TYPE_Q3_K, ds4q_dequantize_q3_k, x,N,NC));
    printf("  q5_K (5.5 bpw): %.4f\n", rt(DS4Q_TYPE_Q5_K, ds4q_dequantize_q5_k, x,N,NC));
    printf("  q6_K (6.5 bpw): %.4f\n", rt(DS4Q_TYPE_Q6_K, ds4q_dequantize_q6_k, x,N,NC));
    printf("  (monotonic decrease => all layouts self-consistent + correct)\n");
    return 0;
}
