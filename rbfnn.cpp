#include<iostream>
#include<cassert>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<vector>
#include<iomanip>

using namespace std;

const int P=100;
vector<double> X(P);
vector<double> Y(P);
const int M=41; // // number of basis function
vector<double> center(M); // basis function center
vector<double> delta(M); // basis function width
vector<double> Weight(M);
const double eta=0.001;
const double ERR=0.9;
const int ITERATION_CEIL=1000;
vector<double> error(P);

inline double funcA(double x){
    return exp(-x)*cos(3*x); // object function
}

inline double uniform(double floor,double ceil){
    return floor+1.0*rand()/RAND_MAX*(ceil-floor);
}

// generate random  normal value
inline double RandomNorm(double mu,double sigma,double floor,double ceil){
    double x,prob,y;
    do{
        x=uniform(floor,ceil);
        prob=1/sqrt(2*M_PI*sigma)*exp(-1*(x-mu)*(x-mu)/(2*sigma*sigma));
        y=1.0*rand()/RAND_MAX;
    }while(y>prob);
    return x;
}

void generateSample(){
    for(double i=0;i<P;++i){
        double in=uniform(-4,4);
        X[i]=in;
        Y[i]=funcA(in)+RandomNorm(0,0.1,-0.3,0.3);
    }
}

void initVector(vector<double> &vec,double floor,double ceil){
    for(int i=0;i<vec.size();++i)
        vec[i]=uniform(floor,ceil);
}

double getOutput(double x){
    double y=0.0;
    for(int i=0;i<M;++i)
        y+=Weight[i]*exp(-1.0*(x-center[i])*(x-center[i])/(2*delta[i]*delta[i]));
    return y;
}

double calSingleError(int index){
    double output=getOutput(X[index]);
    return Y[index]-output;
}

// calculate residual
double calTotalError(){
    double rect=0.0;
    for(int i=0;i<P;++i){
        error[i]=calSingleError(i);
        rect+=error[i]*error[i];
    }
    return rect/2;
}
// update parameters
void updateParam(){
    for(int j=0;j<M;++j){
        double delta_center=0.0,delta_delta=0.0,delta_weight=0.0;
        double sum1=0.0,sum2=0.0,sum3=0.0;
        for(int i=0;i<P;++i){
            sum1+=error[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]))*(X[i]-center[j]);
            sum2+=error[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]))*(X[i]-center[j])*(X[i]-center[j]);
            sum3+=error[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]));
        }
        delta_center=eta*Weight[j]/(delta[j]*delta[j])*sum1;
        delta_delta=eta*Weight[j]/pow(delta[j],3)*sum2;
        delta_weight=eta*sum3;
        center[j]+=delta_center;
        delta[j]+=delta_delta;
        Weight[j]+=delta_weight;
    }
}

int main(){
    srand(time(0));

    initVector(Weight,-0.1,0.1);
    initVector(center,-4.0,4.0);
    initVector(delta,0.1,0.3);

    generateSample();

    int iteration=ITERATION_CEIL;
    while(iteration-->0){
        if(calTotalError()<ERR) // residual < default residual ?
            break;
        updateParam();
    }
    cout<<"Epoch Times:"<<ITERATION_CEIL-iteration-1<<endl;

    for(int i=0;i<100;i++){
        double inp= uniform(-4.0, 4.0);;
        cout<<setprecision(6)<<setiosflags(ios::left);
        cout<<"Iter"<<" ["<<i+1<<"] "<<"Prediction Value:"<<" "<<getOutput(inp)<<" "<<"Actual Value:"<<funcA(inp)<<endl;
	}
	system("pause");
    return 0;
}
