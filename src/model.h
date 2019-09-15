/**
 * Biterm topic model(BTM) with Gbbis sampling 
 * Author: Xiaohui Yan(xhcloud@gmail.com)
 * 2012-9-25
 */
#ifndef _MODEL_H
#define _MODEL_H

#include <vector>
#include <fstream>
#include <cmath>
#include "biterm.h"
#include "doc.h"
#include "pvec.h"
#include "pmat.h"

using namespace std;

class Model
{
public:
  vector<Biterm> bs;

protected:
  int W;      // vocabulary size
  int K;      // number of topics
  int n_iter; // maximum number of iteration of Gibbs Sampling
  int save_step;

  double alpha;  // hyperparameters of p(z)
  double beta;   // hyperparameters of p(w|z)
  double weight; // hyperparameters of weighted decay
  double rho;    // hyperparameters of weighted decay

  // sample recorders
  Pvec<double> nb_z; // n(b|z), size K*1
  Pmat<double> nwz;  // n(w,z), size K*W

  Pvec<double> pw_b; // the background word distribution

  // If true, the topic 0 is set to a background topic that
  // equals to the emperiacal word dsitribution. It can filter
  // out common words
  // メモ: よく出てくるような単語のトピックを0番目に集める
  bool has_background;

public:
  Model(int K, int W, double a, double b, int n_iter, int save_step,
        bool has_b = false) : K(K), W(W), alpha(a), beta(b),
                              n_iter(n_iter), has_background(has_b),
                              save_step(save_step)
  {
    rho = 1.0;
    pw_b.resize(W);
    nwz.resize(K, W);
    nb_z.resize(K);
  }

  // run estimate procedures
  void run(string docs_pt, string res_dir);

private:
  // intialize memeber varibles and biterms
  void model_init(); // load from docs
  void load_docs(string docs_pt);

  // update estimate of a biterm
  void update_biterm(Biterm &bi);

  // reset topic proportions for biterm b
  void reset_biterm_topic(Biterm &bi);

  // assign topic proportions for biterm b
  void assign_biterm_topic(Biterm &bi, vector<double> pz);

  // compute condition distribution p(z|b)
  void compute_pz_b(Biterm &bi, Pvec<double> &p);

  double compute_nu(double t, double tau = 1000, double kappa = 0.8)
  {
    return pow(t + tau, -kappa);
  };

  void reset_rho()
  {
    for (int k = 0; k < K; ++k)
    {
      nb_z[k] *= rho;
      for (int w = 0; w < W; ++w)
      {
        nwz[k][w] *= rho;
      }
    }
    rho = 1.0;
  };

  void save_res(string res_dir);
  void save_pz(string pt);
  void save_pw_z(string pt);
};

#endif
