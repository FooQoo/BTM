#include <cassert>
#include <iostream>
#include <string>
#include <cmath>
#include <limits>
#include <ctime>
#include <algorithm>
#include <numeric>
#include "sampler.h"
#include "str_util.h"
#include "model.h"

void Model::run(string doc_pt, string res_dir)
{
  double M, Mk;
  load_docs(doc_pt);
  model_init();

  cout << "Begin iteration" << endl;
  string out_dir = res_dir + "k" + str_util::itos(K) + ".";
  for (int it = 1; it < n_iter + 1; ++it)
  {
    cout << "\riter " << it << '/' << n_iter;
    fflush(stdout);

    rho *= 1 - compute_nu(it);
    weight = (compute_nu(it) * bs.size()) / rho;

    if (rho < 10e-200)
    {
      reset_rho();
    }
    int b = Sampler::uni_sample(bs.size());
    update_biterm(bs[b]);

    if (it % (int)(n_iter / n_h_opt) == 0)
    {
      reset_rho();
      optimize_alpha();
    }

    if (it % save_step == 0)
      save_res(out_dir);
  }

  reset_rho();

  save_res(out_dir);
}

void Model::model_init()
{
  srand(time(NULL));
  // random initialize
  nb_z.rand_init();
  nwz.rand_init();
}

// input, each line is a doc
// format: wid  wid  wid ...
void Model::load_docs(string dfile)
{
  cout << "load docs: " << dfile << endl;
  ifstream rf(dfile.c_str());
  if (!rf)
  {
    cout << "file not find:" << dfile << endl;
    exit(-1);
  }

  string line;
  while (getline(rf, line))
  {
    Doc doc(line);
    doc.gen_biterms(bs);
    // statistic the exmperial word distribution
    for (int i = 0; i < doc.size(); ++i)
    {
      int w = doc.get_w(i);
      pw_b[w] += 1;
    }
  }

  pw_b.normalize();
}

// sample procedure for ith biterm
void Model::update_biterm(Biterm &bi)
{
  // reset_biterm_topic(bi);

  // compute p(z|b)
  Pvec<double> pz;
  compute_pz_b(bi, pz);

  // update Nk and Nw|k for biterm b
  assign_biterm_topic(bi, pz.to_vector());
}

// compute p(z|w_i, w_j)
void Model::compute_pz_b(Biterm &bi, Pvec<double> &pz)
{
  pz.resize(K);
  int w1 = bi.get_wi();
  int w2 = bi.get_wj();

  double pw1k, pw2k, pk;
  for (int k = 0; k < K; ++k)
  {
    // avoid numerical problem by mutipling W
    if (has_background && k == 0)
    {
      pw1k = pw_b[w1];
      pw2k = pw_b[w2];
    }
    else
    {
      pw1k = (nwz[k][w1] + beta) / (2 * nb_z[k] + W * beta);
      pw2k = (nwz[k][w2] + beta) / (2 * nb_z[k] + 1 + W * beta);
    }
    pk = nb_z[k] + alpha[k];
    pz[k] = pk * pw1k * pw2k;
  }

  pz.normalize();
}

// assign topic k to biterm i
void Model::assign_biterm_topic(Biterm &bi, vector<double> pz)
{
  // bi.set_z(k);
  int w1 = bi.get_wi();
  int w2 = bi.get_wj();

  for (int k = 0; k < K; ++k)
  {
    double pz_k_weighted = pz[k] * weight;
    nb_z[k] += pz_k_weighted;
    nwz[k][w1] += pz_k_weighted;
    nwz[k][w2] += pz_k_weighted;
  }
}

void Model::save_res(string dir)
{
  string pt = dir + "pz";
  cout << "\nwrite p(z): " << pt << endl;
  save_pz(pt);

  string pt2 = dir + "pw_z";
  cout << "write p(w|z): " << pt2 << endl;
  save_pw_z(pt2);
}

// p(z) is determinated by the overall proportions
// of biterms in it
void Model::save_pz(string pt)
{
  Pvec<double> pz(nb_z);
  pz.normalize(alpha);
  pz.write(pt);
}

void Model::save_pw_z(string pt)
{
  Pmat<double> pw_z(K, W); // p(w|z) = phi, size K * M
  ofstream wf(pt.c_str());
  for (int k = 0; k < K; k++)
  {
    for (int w = 0; w < W; w++)
      pw_z[k][w] = (nwz[k][w] + beta) / (nb_z[k] * 2 + W * beta);

    wf << pw_z[k].str() << endl;
  }
}
