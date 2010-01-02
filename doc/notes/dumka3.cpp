#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
using namespace std;
/* VectorFunctionBase - interface with three functions, which have to be implemented
 *        void  compute(Vector& y, double t, Vector& result) - calculates right hand side of the ODE y'=f(y,t)
 *        double  cour(Vector& y, double t) - used if "isComputeEigenValue=false" to
 *                                                calculate h_{euler} - step which you would use in explicit Euler method
 *        Element GetNonZeroElement() - can be used if "isComputeEigenValue=true" in eigenvalue function, if
 *                                          iterations of power method have to be restarted, but y,f(y) equals to zero,
 *                                          in this case program need to initialize random non-zero vector, because
 *                                          vector's Element is unknown in advance
 *                                          for example if Vector contains "double"(s) you can return 0.01*(rand()%100)
 *                                          but if Vector contains "Point"(s) you can
 *                                          return  Point(0.01*(rand()%100),0.01*(rand()%100),0.01*(rand()%100))
 */
template <class Vector, class Element>
class VectorFunctionBase {
  public:
    virtual void compute(Vector& y, double t, Vector& result) = 0;
    virtual double cour(Vector& y, double t) = 0;
    virtual Element GetNonZeroElement() = 0;
};

/*
 *  dumka3 - solves non-linear mildly - stiff system of ordinary differential equations y'=f(y,t)
 *           in order to be able to use this function you should implement interface VectorFunctionBase (above)
 *           t      - time
 *           tend   - end of the integration interval
 *           h0     - initial step-size (this value can be override by the program)
 *           atol    - tolerance of the computation (should be strictly positive)
 *           rtol - relative tolerance (should be strictly positive)
 *           f      - Class which implement VectorFunctionBase interface, or class , which has 3 functions
 *                    compute(Vector& y, double t, Vector& result) - calculates right hand side
 *                    cour(Vector& y, double t) - used if "isComputeEigenValue=false" to
 *                                                calculate h_{euler} - step which you would use in explicit Euler method
 *                    GetNonZeroElement() - can be used if "isComputeEigenValue=true" in eigenvalue function, if
 *                                          iterations of power method have to be restarted, but y,f(y) equals to zero,
 *                                          in this case program need to initialize random non-zero vector, because
 *                                          vector's Element can be complicated like Point etc, user should provide random non-zero element
 *           y     - solution of the ODE y'=f(y,t)
 *           isComputeEigenValue - if "true": function "eigenvalue" will be used to determine maximum eigenvalue and h_{euler}=2/eigenvalue
 *                               - if "false": user should provide H_{euler} via "cour" function of the VectorFunction f
 *
 */
template <class VectorFunction, class Vector>
void dumka3(double& t, const double& tend, const double& h0,const double& atol,
    const double& rtol, VectorFunction& f, Vector& y, bool isComputeEigenValue = false ) {
  static const long _n_p=14;
  static const long _sz=2400;
  static const long _sz2=6;
  long numOfRejectedSteps=0;
  double minimumValue = 1.e-15;
  double err_n=0.;
  double h_n=0;
  long numOfSteps=0;
  double maxCou = 0.;
  double meanCou = 0.;
  double timeInterval = tend - t;
  if (timeInterval<h0) {
    cout << "Initial step size h0 is too small: t+h0>tend" << endl;
    return;
  };
  if ( atol < minimumValue ) {
    cout << "atol must be positive" << endl;
    return;
  }
  if ( rtol < minimumValue ) {
    cout << "rtol must be positive" << endl;
    return;
  }
  long numOfRHSEvaluations=0;
  long numOfRHSEvaluations0;
  double c_2,c_3,c_4,a_21,a_31,a_41,a_32,a_42,a_43,t2,t3,t4,tmp_1,tmp_2;
  double r,cou,h_new,t_en,h;
  int n_deg[_n_p]={3,6,9,15,21,27,36,48,63,81,135,189,243,324};
  int index_first[_n_p]={1,2,4,7,12,19,28,40,56,77,104,149,212,293};
  int index_last[_n_p]={1,3,6,11,18,27,39,55,76,103,148,211,292,400};
  const double stab_reg[_n_p]={ 2.5005127005e+0, 9.784756428464169e+0,
    23.818760475282560e+0, 68.932817124349140e+0, 136.648186730571300e+0,
    226.8897061613812e+0, 404.7232537379578e+0, 720.7401653373073e+0,
    1337.1268312643200e+0, 3266.0271134242240e+0, 9072.0255133253400e+0,
    17784.544093341440e+0, 29376.4540800858800e+0,50666.52463738415826e+0};

  const double coef[_sz];


  long j;
  long _size=y.size();
  Vector z0(_size), z1(_size), z2(_size), z3(_size), oldEigenVector(_size);
  double lambdaOld = 0.;
  //Vector z0, z1, z2, z3;
  if ( h0 < 1.e-14 ) {
    cout << "Initial step h0 is too small" << endl;
    return;
  }
  if ( tend <= t ) {
    cout << "End-time Tend is less than initial time" << endl;
    return;
  }
  h_new=h0;
  //      z0=f.compute(y,t);
  f.compute(y,t,z0);
  if ( isComputeEigenValue ) {
    f.compute(z0,t,oldEigenVector);
    numOfRHSEvaluations++;
  }
  numOfRHSEvaluations++;
  long stepId=0;
  bool stepOK = true;
  do {
    if ( isComputeEigenValue ) {//calculate eigen value only every 20th step
      if ( stepId%20 == 0 || !stepOK ) {
        double ev = abs( eigenvalue(y, z0, t, f, oldEigenVector, z1, lambdaOld,
              numOfRHSEvaluations,minimumValue) );
        if ( ev < minimumValue ) {
          ev = minimumValue;
        }
        cou = 2./ (ev*1.2);
      }
    } else {
      cou = f.cour (y, t);
    }
    if ( maxCou < cou ) {
      maxCou=cou;
    };
    meanCou += cou;
    numOfSteps++;
    h=min(h_new,(tend-t));
    // find degree of the polynomials to be used
    int index=0;
    while( (index!=_n_p-1) && ((2.e+0*h/cou)>stab_reg[index]) ) {
      index++;
    };

    if(index>0) {
      if(((stab_reg[index-1]/((double) n_deg[index-1]))*((double) n_deg[index]))>(2.*h/cou)) {
        index=index-1;
      };
    };
    int n_pol_degree=n_deg[index];
    cout << "Time= " << t << " Degree of the polynomial=" <<  n_pol_degree << endl;

    h=min((stab_reg[index]*cou/2.e+0),h);

    long n_pol=0;
    numOfRHSEvaluations0 = numOfRHSEvaluations;
    for (int k=index_first[index]-1; k < index_last[index]; k++ ) {
      // save initial conditions (solution, right hand side, time)
      // that will be needed if time step will be rejected
      if(k==index_first[index]-1) {
        for ( j=0; j<_size; j++ ) {
          z2[j]=z0[j]; //z2=z_en
          z3[j]=y[j]; //z3=y_en
        };
        t_en=t;
      };
      n_pol=n_pol+3;
      int _idx=_sz2*k;
      a_21=h*coef[_idx];
      c_2=a_21;
      a_31=h*coef[_idx+1];
      a_32=h*coef[_idx+2];
      c_3=a_31+a_32;
      a_41=h*coef[_idx+3];
      a_42=h*coef[_idx+4];
      a_43=h*coef[_idx+5];
      c_4=a_41+a_42+a_43;
      //long _size=y.size();
      for ( j=0; j<_size; j++ ) {
        y[j]=y[j]+a_21*z0[j];
      };
      t2=t+c_2;

      // marker W ******************************
      //      z1=f.compute(y,t2);
      f.compute(y,t2,z1);
      numOfRHSEvaluations++;
      if(n_pol==n_deg[index]) {
        r=h*(coef[_idx+1]-coef[_idx]);
        for ( j=0; j<_size; j++ ) {
          y[j]=y[j]+r*z0[j]+a_32*z1[j];
        };
      } else {
        for ( j=0; j<_size; j++ ) {
          y[j]=y[j]+a_32*z1[j];
        };
      };

      // marker X ******************************
      t3=t+c_3;
      if(n_pol==n_deg[index]) {
        tmp_1=c_3/2.e+0;
        tmp_2=(c_4-c_2)/2.e+0;
        for ( j=0; j<_size; j++ ) {
          z1[j]=tmp_1*z1[j]-tmp_2*z0[j];
        };
      };

      // marker Y ******************************
      //      z0 = f.compute(y,t3);
      f.compute(y,t3,z0);
      numOfRHSEvaluations++;
      if(n_pol==n_deg[index]) {
        for ( j=0; j<_size; j++ ) {
          z1[j]=tmp_2*z0[j]+z1[j];
        };
      };
      // marker Z ******************************
      for ( j=0; j<_size; j++ ) {
        y[j]=y[j]+a_43*z0[j];
      };

      t4=t+c_4;
      t=t4;
      //     z0=f.compute(y,t);
      f.compute(y,t,z0);
      numOfRHSEvaluations++;
    };

    //long _size=z0.size();
    for ( j=0; j < _size; j++ ) {
      z1[j]=(1./(rtol*max(sqrt(y[j]*y[j]),sqrt(z3[j]*z3[j]))+atol))*((-tmp_1)*z0[j]+z1[j]);
    };
    h_new=h;
    // check error and return new recommended step size in variable h_new

    if(!isStepAccepted(t,y,z1,h_new,n_pol_degree,err_n, h_n, stepId)) {
      numOfRejectedSteps = numOfRejectedSteps + (numOfRHSEvaluations - numOfRHSEvaluations0);
      stepOK = false;
      f.isStepAccepted = false;
      //long _size=y.size();
      /* code for paraller exceution via OpemMP - you may want to use it
#pragma parallel shared(y,z0,z2,z3) local(j)
#pragma pfor
*/
      for ( j=0; j < _size; j++ ) {
        z0[j]=z2[j];
        y[j]=z3[j];
      };
      t=t_en;
      cout << "Step is rejected. Used degree of polynomial: " << n_pol_degree << " Time= " << t << endl;
    } else {
      stepOK = true;
      f.isStepAccepted = true;
    }
    stepId++;
  } while ( t < tend );
  meanCou = meanCou/((double)numOfSteps);
  cout << "Number of RHS evaluations = " << numOfRHSEvaluations << endl;
  cout << "Mean step-size = " << timeInterval/((double)numOfRHSEvaluations) << endl;
  cout << "Mean value of cou = " << meanCou << endl;
  // cout << "Maximum value of cou = " << maxCou<< endl;
  cout << "Last value of cou = " << cou << endl;
  cout << "Number of rejected RHS evaluations = " << numOfRejectedSteps << endl;
  //delete [] z;
};

/* norm1 - function isStepAccepted can use different norms, but we implemente "max=max|u_i|, i= 1..n" norm
 *         one can optimized this function by replacing qrt(z[i]*z[i]) to abs(z[i]), but you must define
 *         function abs for Element of the Vector
 */
template<class Vector>
double norm1 (const Vector& z) {
  double eps1=0.e+0;
  long _size=z.size();
  for ( int i=0; i < _size; i++ ) {
    double absValue = sqrt(z[i]*z[i]);
    if(eps1<absValue) eps1=absValue;
  };
  return eps1;
};

/* isStepAccepted - return true if step can be accepted, and at the same time this function
 *                  calculates and return h_new
 *
 */
template<class Vector>
bool  isStepAccepted(const double &t, const Vector &y, const Vector &z,
    double &h_new, const int &n_pol_degree, double err_n, double h_n, long stepId) {

  bool isnotreject = false;
  double eps1=0.e0;
  long size = y.size();
  for ( int i=0;i < size; i++ ) {
    eps1=eps1+z[i]*z[i];
  }
  double eps=sqrt( eps1/((double)size) );
  double fracmin=0.1e0;
  if(eps==0.e0) {
    eps=1.e-14;
  }
  double frac=pow(1.e0/eps,1.e0/3.e0);
  if(eps <= 1.e0) {
    if (err_n>0.e0 && h_n > 0.e0) {
      double frac2=pow(err_n,(1.e0/3.e0))*frac*frac*(h_new/h_n);
      frac=min(frac,frac2);
    }
    isnotreject = true;
    double fracmax=2.e0;
    frac=min(fracmax,max(fracmin,0.8e0*frac));
    double h_old = h_new;
    h_new=frac*h_new;
    h_n=h_old;
    err_n=eps;
  } else {
    isnotreject = false;
    double fracmax=1.e0;
    frac=0.8e0*min(fracmax,max(fracmin,0.8e0*frac));
    if(stepId==0) {
      h_new=fracmin*h_new;
    } else {
      h_new=frac*h_new;
    }
    cout << eps << t << n_pol_degree << h_new << h_n << endl;
  }
  return isnotreject;
};

/* norm - calculates norm of the Vector, this function works only if Element of
 *        the Vector has operation multiplication "*"
 */
template <class Vector>
double norm ( const Vector &x ) {
  int size = x.size();
  double norm = 0.;
  for ( int i=0; i < size; i++ ) {
    norm += x[i]*x[i];
  }
  norm = sqrt(norm);
  return norm;
}

/* eigenvalue - calculates eigen values of the right hand side
 *              The used algorithm is a slight modification of the algorithm
 *              used in ROCK and RKC
 */
  template <class VectorFunction, class Vector>
double eigenvalue(const Vector& y,Vector& fy, double t,
    VectorFunction& f, Vector& v, Vector& result,
    double& lambdaOld, long& numOfRHSEvaluations, double minimumValue)
{
  cout << "Compute eigenvalue ..." << endl;
  int maxit = 30;
  double tol = 0.01;
  int idx = 0;
  int size = y.size();
  double radius = 1.e-10;
  double lambda = lambdaOld;
  double nrmY = norm(y);
  double  nrmV = norm(v);
  //f.compute(result,t,v);
  //f.compute(fy,t,v);
  do {
    idx++;
    // all vectors should be ||y-v|| < radius, because problem can be non-linear
    if ( nrmV < minimumValue )
    {
      // change initial vector and restart - if restarted check that
      // initial vector has not been used already
      for ( int i=0; i < size; i++ )
      {
        v[i] = f.GetNonZeroElement();
      };
      nrmV = norm(v);
      if ( nrmV <   minimumValue ) {
        cout << "Error occur: your GetNonZeroElement function should return a value with strictly positive norm, but it returns 0!" << endl;
        cout << "Fix GetNonZeroElement function, please." << endl;
        return lambdaOld;
      }
    }
    double norminv= radius/nrmV;
    for ( int i=0; i < size; i++ ) {
      v[i] = y[i] + norminv * v[i];
    }

    f.compute(v,t,result);
    numOfRHSEvaluations++;
    // approximation of eigenvector
    for ( int i=0; i < size; i++ ) {
      v[i] = result[i] - fy[i];
    }
    // approximation of eigenvalue
    nrmV = norm(v);
    lambdaOld = lambda;
    lambda = nrmV/radius;
    cout << "Iteration = " << idx << " eigenvalue = " << lambda << endl;
  }
  while ( abs(lambda - lambdaOld) > abs(lambda)*tol && idx <= maxit );
  lambdaOld = lambda;
  return lambda;
}
